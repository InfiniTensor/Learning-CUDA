#include "include/fused_hadamard_quant.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)
using fused_hq::DType; using fused_hq::Version;

struct Options {
    int batch = 32, dim = 1024, warmup = 100, repeat = 1000, trials = 3;
    std::string dtype = "fp16", version = "all", csv, pattern = "random";
    bool correctness_only = false, profile = false;
};
struct Errors { int max_q = 0; float max_amax = 0, max_scale = 0; bool correct = false; };
struct Measurement { Version version; std::vector<float> samples; float median = 0, minimum = 0, maximum = 0; Errors errors; };

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; auto value = [&]() { if (++i == argc) throw std::runtime_error("missing value for " + a); return std::string(argv[i]); };
        if (a == "--batch") o.batch = std::stoi(value()); else if (a == "--dim") o.dim = std::stoi(value());
        else if (a == "--warmup") o.warmup = std::stoi(value()); else if (a == "--repeat") o.repeat = std::stoi(value());
        else if (a == "--trials") o.trials = std::stoi(value()); else if (a == "--dtype") o.dtype = value();
        else if (a == "--version" || a == "--kernel") o.version = value(); else if (a == "--csv") o.csv = value();
        else if (a == "--pattern") o.pattern = value(); else if (a == "--correctness-only") o.correctness_only = true;
        else if (a == "--profile") o.profile = true; else if (a == "--help") {
            std::cout << "--version v0|v1|v2|separate|all --batch N --dim D --dtype fp32|fp16|bf16|all\n"
                         "--warmup N --repeat N --trials N --pattern random|basis|sequence|zero --profile --correctness-only --csv PATH\n"; std::exit(0);
        } else throw std::runtime_error("unknown option: " + a);
    }
    if (o.profile) { o.repeat = std::min(o.repeat, 3); o.trials = 1; }
    return o;
}
static std::vector<Version> versions(const std::string& s) {
    if (s == "all") return {Version::kV0, Version::kV1, Version::kV2, Version::kSeparate};
    if (s == "v0" || s == "fused_v0") return {Version::kV0}; if (s == "v1" || s == "fused_v1") return {Version::kV1};
    if (s == "v2" || s == "fused_v2") return {Version::kV2}; if (s == "separate") return {Version::kSeparate};
    throw std::runtime_error("invalid version: " + s);
}
static std::vector<DType> dtypes(const std::string& s) {
    if (s == "all") return {DType::kFloat32, DType::kFloat16, DType::kBFloat16};
    if (s == "fp32") return {DType::kFloat32}; if (s == "fp16") return {DType::kFloat16}; if (s == "bf16") return {DType::kBFloat16};
    throw std::runtime_error("invalid dtype: " + s);
}
static std::vector<float> make_input(int batch, int dim, const std::string& pattern) {
    std::vector<float> x(static_cast<std::size_t>(batch) * dim); std::mt19937 rng(20260920); std::uniform_real_distribution<float> dist(-1, 1);
    for (int r = 0; r < batch; ++r) for (int c = 0; c < dim; ++c) {
        if (pattern == "random") x[static_cast<std::size_t>(r) * dim + c] = dist(rng);
        else if (pattern == "basis") x[static_cast<std::size_t>(r) * dim + c] = c == (r % dim) ? 1.0f : 0.0f;
        else if (pattern == "sequence") x[static_cast<std::size_t>(r) * dim + c] = static_cast<float>((c % 31) - 15) / 16.0f;
        else if (pattern == "zero") x[static_cast<std::size_t>(r) * dim + c] = 0.0f; else throw std::runtime_error("invalid pattern");
    } return x;
}
template <typename T> static T cast_input(float x);
template <> float cast_input<float>(float x) { return x; }
template <> __half cast_input<__half>(float x) { return __float2half(x); }
template <> __nv_bfloat16 cast_input<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template <typename T> static float cast_back(T x);
template <> float cast_back<float>(float x) { return x; }
template <> float cast_back<__half>(__half x) { return __half2float(x); }
template <> float cast_back<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

static void reference(std::vector<float> values, int batch, int dim, std::vector<std::int8_t>& q, std::vector<float>& amax, std::vector<float>& scale) {
    q.resize(values.size()); amax.resize(batch); scale.resize(batch); const float norm = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int r = 0; r < batch; ++r) {
        float* row = values.data() + static_cast<std::size_t>(r) * dim;
        for (int stride = 1; stride < dim; stride <<= 1) for (int base = 0; base < dim; base += 2 * stride)
            for (int j = 0; j < stride; ++j) { float x = row[base+j], y = row[base+j+stride]; row[base+j] = x+y; row[base+j+stride] = x-y; }
        float m = 0; for (int c = 0; c < dim; ++c) { row[c] *= norm; m = std::max(m, std::fabs(row[c])); }
        amax[r] = m; scale[r] = m > 0 ? m / 127.0f : 1.0f;
        for (int c = 0; c < dim; ++c) { int z = static_cast<int>(std::nearbyint(row[c] / scale[r])); q[static_cast<std::size_t>(r)*dim+c] = static_cast<std::int8_t>(std::max(-127, std::min(127, z))); }
    }
}

template <typename T> static std::vector<Measurement> run(const Options& o, DType dtype) {
    std::vector<float> source = make_input(o.batch, o.dim, o.pattern), actual(source.size()); std::vector<T> host(source.size());
    for (std::size_t i=0;i<source.size();++i) { host[i]=cast_input<T>(source[i]); actual[i]=cast_back<T>(host[i]); }
    std::vector<std::int8_t> ref_q; std::vector<float> ref_a, ref_s; reference(actual,o.batch,o.dim,ref_q,ref_a,ref_s);
    T* input=nullptr; std::int8_t* output=nullptr; float *amax=nullptr,*scale=nullptr; void* workspace=nullptr;
    CUDA_CHECK(cudaMalloc(&input,host.size()*sizeof(T))); CUDA_CHECK(cudaMalloc(&output,host.size()));
    CUDA_CHECK(cudaMalloc(&amax,o.batch*sizeof(float))); CUDA_CHECK(cudaMalloc(&scale,o.batch*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&workspace,host.size()*sizeof(float))); CUDA_CHECK(cudaMemcpy(input,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::vector<Measurement> result;
    for (Version v: versions(o.version)) {
        Measurement m; m.version=v; CUDA_CHECK(fused_hq::launch(v,input,output,amax,scale,workspace,o.batch,o.dim,dtype)); CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::int8_t> got_q(host.size()); std::vector<float> got_a(o.batch),got_s(o.batch);
        CUDA_CHECK(cudaMemcpy(got_q.data(),output,got_q.size(),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(got_a.data(),amax,o.batch*sizeof(float),cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(got_s.data(),scale,o.batch*sizeof(float),cudaMemcpyDeviceToHost));
        for (std::size_t i=0;i<got_q.size();++i) m.errors.max_q=std::max(m.errors.max_q,std::abs(int(got_q[i])-int(ref_q[i])));
        for (int i=0;i<o.batch;++i) { m.errors.max_amax=std::max(m.errors.max_amax,std::fabs(got_a[i]-ref_a[i])); m.errors.max_scale=std::max(m.errors.max_scale,std::fabs(got_s[i]-ref_s[i])); }
        float tol = dtype==DType::kFloat32 ? 2e-5f : (dtype==DType::kFloat16 ? 2e-3f : 2e-2f);
        m.errors.correct=m.errors.max_q<=1 && m.errors.max_amax<=tol && m.errors.max_scale<=tol/127.0f;
        if (!o.correctness_only) {
            for(int i=0;i<o.warmup;++i) CUDA_CHECK(fused_hq::launch(v,input,output,amax,scale,workspace,o.batch,o.dim,dtype)); CUDA_CHECK(cudaDeviceSynchronize());
            for(int trial=0;trial<o.trials;++trial) { cudaEvent_t begin,end; CUDA_CHECK(cudaEventCreate(&begin)); CUDA_CHECK(cudaEventCreate(&end));
                if(o.profile) CUDA_CHECK(cudaProfilerStart()); CUDA_CHECK(cudaEventRecord(begin));
                for(int i=0;i<o.repeat;++i) CUDA_CHECK(fused_hq::launch(v,input,output,amax,scale,workspace,o.batch,o.dim,dtype));
                CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end)); if(o.profile) CUDA_CHECK(cudaProfilerStop());
                float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end)); m.samples.push_back(ms*1000.0f/o.repeat); cudaEventDestroy(begin); cudaEventDestroy(end);
            }
            auto sorted=m.samples; std::sort(sorted.begin(),sorted.end()); m.median=sorted[sorted.size()/2]; m.minimum=sorted.front(); m.maximum=sorted.back();
        }
        result.push_back(m);
    }
    cudaFree(workspace); cudaFree(scale); cudaFree(amax); cudaFree(output); cudaFree(input); return result;
}

static bool emit(const Options& o, DType dtype, const std::vector<Measurement>& values) {
    float separate=0; for(auto& m:values) if(m.version==Version::kSeparate) separate=m.median;
    std::ostream* out=&std::cout; std::ofstream file; bool header=false;
    if(!o.csv.empty()) { std::ifstream test(o.csv); header=test.peek()==std::ifstream::traits_type::eof(); test.close(); file.open(o.csv,std::ios::app); out=&file; if(header) *out<<"implementation,batch,head_dim,dtype,trial,latency_us,median_us,min_us,max_us,variation,speedup_vs_separate,latency_ratio_vs_separate,percent_latency_reduction,correct,max_quant_error,max_amax_error,max_scale_error,pattern\n"; }
    if(o.csv.empty()) std::cout<<"Shape: ["<<o.batch<<", "<<o.dim<<"] dtype: "<<fused_hq::dtype_name(dtype)<<" pattern: "<<o.pattern<<"\n";
    bool all_correct=true; for(auto& m:values) {
        all_correct = all_correct && m.errors.correct;
        float variation=m.median>0?(m.maximum-m.minimum)/m.median:0, speed=(separate>0&&m.median>0)?separate/m.median:0, ratio=(separate>0)?m.median/separate:0;
        if(o.csv.empty()) std::cout<<std::left<<std::setw(12)<<fused_hq::version_name(m.version)<<" correct="<<(m.errors.correct?"PASS":"FAIL")<<" q_err="<<m.errors.max_q<<" amax_err="<<m.errors.max_amax<<(o.correctness_only?"":" latency="+std::to_string(m.median)+" us")<<"\n";
        else for(std::size_t i=0;i<std::max<std::size_t>(1,m.samples.size());++i) *out<<fused_hq::version_name(m.version)<<','<<o.batch<<','<<o.dim<<','<<fused_hq::dtype_name(dtype)<<','<<i<<','<<(m.samples.empty()?0:m.samples[i])<<','<<m.median<<','<<m.minimum<<','<<m.maximum<<','<<variation<<','<<speed<<','<<ratio<<','<<(1-ratio)*100<<','<<(m.errors.correct?"true":"false")<<','<<m.errors.max_q<<','<<m.errors.max_amax<<','<<m.errors.max_scale<<','<<o.pattern<<'\n';
    } return all_correct;
}
int main(int argc,char** argv) { try { Options o=parse(argc,argv); if(o.dim<32||o.dim>8192||(o.dim&(o.dim-1))) throw std::runtime_error("dim must be a power of two in [32,8192]");
    bool correct=true; for(DType d:dtypes(o.dtype)) { if(d==DType::kFloat32) correct &= emit(o,d,run<float>(o,d)); else if(d==DType::kFloat16) correct &= emit(o,d,run<__half>(o,d)); else correct &= emit(o,d,run<__nv_bfloat16>(o,d)); }
    return correct ? 0 : 2; } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; } }
