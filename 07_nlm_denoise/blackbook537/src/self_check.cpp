// ============================================================================
// units 子命令：Host 侧单元测试（不依赖 GPU，可在任意构建环境运行）
// 覆盖：参数解析、质量指标、CPU 参考实现的关键不变式
// ============================================================================
#include <cmath>
#include <cstdio>
#include <string>

#include "nlm_cpu_ref.h"
#include "params.h"
#include "benchmark.h"

namespace {

int g_failures = 0;

void Check(bool cond, const char* name) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) ++g_failures;
}

// ---- 参数解析 ----
void TestParams() {
    std::printf("[units] 参数解析\n");
    const char* path = "output/work/units_params_tmp.txt";
    EnsureParentDir(path);
    {
        FILE* fp = std::fopen(path, "w");
        std::fprintf(fp,
                     "# 注释行\n"
                     "patch_radius = 3\n"
                     "search_radius = 10   # 行内注释\n"
                     "h = 12.5\n"
                     "sigma = 30.0\n");
        std::fclose(fp);
    }
    NlmParams p;
    std::string err;
    Check(ParseParams(path, &p, &err) &&
          p.patch_radius == 3 && p.search_radius == 10 &&
          p.h == 12.5f && p.sigma == 30.0f,
          "正常解析（含注释/缺省）");

    {  // 未知 key 必须报错
        FILE* fp = std::fopen(path, "w");
        std::fprintf(fp, "unknown_key = 1\n");
        std::fclose(fp);
        Check(!ParseParams(path, &p, &err), "未知 key 拒绝");
    }
    {  // 越界取值必须报错
        FILE* fp = std::fopen(path, "w");
        std::fprintf(fp, "patch_radius = 0\n");
        std::fclose(fp);
        Check(!ParseParams(path, &p, &err), "越界取值拒绝");
    }
    std::remove(path);
}

// ---- 质量指标 ----
void TestMetrics() {
    std::printf("[units] 质量指标\n");
    ImageU8 a = MakeSyntheticImage(16, 16, 1, 0.0f, 42u);
    ImageU8 b = a;
    Check(ComputeMAE(a, b) == 0.0, "MAE(相同图像)=0");
    Check(std::isinf(ComputePSNR(a, b)), "PSNR(相同图像)=inf");
    for (size_t i = 0; i < b.data.size(); ++i) b.data[i] = (uint8_t)(a.data[i] + 10);
    Check(std::fabs(ComputeMAE(a, b) - 10.0) < 1e-9, "MAE(恒差10)=10");
    // PSNR = 20*log10(255/10) ≈ 28.13 dB
    Check(std::fabs(ComputePSNR(a, b) - 28.1308) < 0.01, "PSNR(恒差10)≈28.13dB");
}

// ---- CPU 参考实现不变式 ----
void TestCpuRef() {
    std::printf("[units] CPU 参考实现\n");
    NlmParams p;
    p.patch_radius = 2;
    p.search_radius = 3;

    {  // 常量图：所有 patch 距离为 0，权重全 1，输出必须逐位等于输入
        ImageU8 img;
        img.width = 48; img.height = 48; img.channels = 1;
        img.data.assign(48 * 48, 128);
        ImageU8 out;
        NlmDenoiseCpuRef(img, &out, p);
        Check(out.data == img.data, "常量灰度图恒等");
    }
    {  // RGB 常量图
        ImageU8 img;
        img.width = 32; img.height = 32; img.channels = 3;
        img.data.resize(32 * 32 * 3);
        for (size_t i = 0; i < img.data.size(); i += 3) {
            img.data[i] = 100; img.data[i + 1] = 150; img.data[i + 2] = 200;
        }
        ImageU8 out;
        NlmDenoiseCpuRef(img, &out, p);
        Check(out.data == img.data, "常量 RGB 图恒等");
    }
    {  // 确定性：同一输入两次运行结果 bit 级一致
        ImageU8 img = MakeSyntheticImage(64, 64, 1, 25.0f, 7u);
        ImageU8 o1, o2;
        NlmDenoiseCpuRef(img, &o1, p);
        NlmDenoiseCpuRef(img, &o2, p);
        Check(o1.data == o2.data, "确定性（两次运行一致）");
    }
    {  // 极小图（图像小于搜索窗）：clamp 边界不越界、输出范围合法
        ImageU8 img = MakeSyntheticImage(5, 7, 1, 25.0f, 9u);
        NlmParams big;
        big.patch_radius = 3;
        big.search_radius = 10;
        ImageU8 out;
        NlmDenoiseCpuRef(img, &out, big);
        bool range_ok = (out.data.size() == img.data.size());
        Check(range_ok, "极小图（5x7, sr=10）不崩溃且尺寸正确");
    }
    {  // 降噪有效性：含噪渐变图降噪后 MAE(输出, 干净图) 应小于 MAE(含噪图, 干净图)
        ImageU8 clean = MakeSyntheticImage(64, 64, 1, 0.0f, 5u);
        ImageU8 noisy = MakeSyntheticImage(64, 64, 1, 25.0f, 5u);
        NlmParams q;  // 基线参数
        ImageU8 out;
        NlmDenoiseCpuRef(noisy, &out, q);
        const double mae_before = ComputeMAE(noisy, clean);
        const double mae_after = ComputeMAE(out, clean);
        std::printf("    降噪前后对干净图 MAE: %.2f -> %.2f\n", mae_before, mae_after);
        Check(mae_after < mae_before, "降噪有效性（MAE 下降）");
    }
}

} // namespace

int RunUnitTests() {
    std::printf("========== NLM 单元测试 ==========" "\n");
    TestParams();
    TestMetrics();
    TestCpuRef();
    std::printf("========== %s（失败 %d 项）==========\n",
                g_failures == 0 ? "全部通过" : "存在失败", g_failures);
    return g_failures;
}

#ifdef NLM_UNITS_STANDALONE
int main() { return RunUnitTests(); }
#endif
