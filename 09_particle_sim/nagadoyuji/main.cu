/**
 * Particle motion simulation in uniform magnetic field (CUDA).
 * Reads particle init, B-field, and params; runs numerical integration on GPU;
 * records trajectory at intervals and writes binary output for Python visualization.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

// -----------------------------------------------------------------------------
// Config and data (host-side)
// -----------------------------------------------------------------------------
struct SimParams {
    double dt = 1e-10;
    int num_steps = 10000;
    int record_interval = 100;
};

// SoA on host: N particles
std::vector<float> h_pos;   // 3*N
std::vector<float> h_vel;   // 3*N
std::vector<float> h_q;     // N
std::vector<float> h_m;     // N
float h_B[3] = {0, 0, 1.0f};
int N = 0;
bool use_B_grid = false;
int g_nx = 0, g_ny = 0, g_nz = 0;
float g_ox = 0, g_oy = 0, g_oz = 0, g_dx = 1, g_dy = 1, g_dz = 1;
std::vector<float> h_Bgrid;  // 3 * nx * ny * nz

SimParams params;

// -----------------------------------------------------------------------------
// Input parsing: particles, uniform/grid B-field, simulation params
// -----------------------------------------------------------------------------
bool load_particles(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        float x, y, z, vx, vy, vz, q, m;
        if (sscanf(line.c_str(), "%f %f %f %f %f %f %f %f", &x, &y, &z, &vx, &vy, &vz, &q, &m) != 8)
            continue;
        h_pos.push_back(x); h_pos.push_back(y); h_pos.push_back(z);
        h_vel.push_back(vx); h_vel.push_back(vy); h_vel.push_back(vz);
        h_q.push_back(q);   h_m.push_back(m);
    }
    N = (int)h_q.size();
    return N > 0;
}

bool load_field(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    std::string line;
    if (!std::getline(f, line)) return false;
    if (line.find("grid") != std::string::npos) {
        use_B_grid = true;
        std::string grid_path;
        if (!std::getline(f, grid_path)) return false;
        while (!grid_path.empty() && (grid_path.back() == '\r' || grid_path.back() == ' ')) grid_path.pop_back();
        FILE* bin = fopen(grid_path.c_str(), "rb");
        if (!bin) { fprintf(stderr, "Cannot open grid %s\n", grid_path.c_str()); return false; }
        int nx, ny, nz;
        if (fread(&nx, sizeof(int), 1, bin) != 1 || fread(&ny, sizeof(int), 1, bin) != 1 || fread(&nz, sizeof(int), 1, bin) != 1) {
            fclose(bin); return false;
        }
        float ox, oy, oz, dx, dy, dz;
        if (fread(&ox, sizeof(float), 1, bin) != 1 || fread(&oy, sizeof(float), 1, bin) != 1 || fread(&oz, sizeof(float), 1, bin) != 1 ||
            fread(&dx, sizeof(float), 1, bin) != 1 || fread(&dy, sizeof(float), 1, bin) != 1 || fread(&dz, sizeof(float), 1, bin) != 1) {
            fclose(bin); return false;
        }
        g_nx = nx; g_ny = ny; g_nz = nz;
        g_ox = ox; g_oy = oy; g_oz = oz; g_dx = dx; g_dy = dy; g_dz = dz;
        size_t count = (size_t)nx * ny * nz * 3;
        h_Bgrid.resize(count);
        if (fread(h_Bgrid.data(), sizeof(float), count, bin) != count) { fclose(bin); return false; }
        fclose(bin);
        return true;
    }
    if (sscanf(line.c_str(), "%f %f %f", &h_B[0], &h_B[1], &h_B[2]) != 3) return false;
    return true;
}

bool load_params(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    std::string line;
    while (std::getline(f, line)) {
        size_t i = line.find('#');
        if (i != std::string::npos) line.resize(i);
        double v;
        if (sscanf(line.c_str(), " dt = %lf", &v) == 1) params.dt = v;
        else if (sscanf(line.c_str(), " num_steps = %d", &params.num_steps) == 1) {}
        else if (sscanf(line.c_str(), " record_interval = %d", &params.record_interval) == 1) {}
    }
    return true;
}

// -----------------------------------------------------------------------------
// CUDA kernels: Boris push (magnetic field only), then position update
// -----------------------------------------------------------------------------
__device__ void cross(float ax, float ay, float az, float bx, float by, float bz,
                      float* ox, float* oy, float* oz) {
    *ox = ay * bz - az * by;
    *oy = az * bx - ax * bz;
    *oz = ax * by - ay * bx;
}

__global__ void integrate_kernel(float* pos, float* vel, const float* q, const float* m,
                                 float Bx, float By, float Bz, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float qm = q[i] / m[i];
    float half_dt = dt * 0.5f;
    // t = (q/m) * B * (dt/2)
    float tx = qm * Bx * half_dt;
    float ty = qm * By * half_dt;
    float tz = qm * Bz * half_dt;

    float vx = vel[i * 3 + 0], vy = vel[i * 3 + 1], vz = vel[i * 3 + 2];
    // v_prime = v + v x t
    float vpx, vpy, vpz;
    cross(vx, vy, vz, tx, ty, tz, &vpx, &vpy, &vpz);
    vpx += vx; vpy += vy; vpz += vz;
    // s = 2*t / (1 + t·t)
    float t2 = tx*tx + ty*ty + tz*tz;
    float sx = 2.f * tx / (1.f + t2);
    float sy = 2.f * ty / (1.f + t2);
    float sz = 2.f * tz / (1.f + t2);
    // v_plus = v + v_prime x s
    float vqx, vqy, vqz;
    cross(vpx, vpy, vpz, sx, sy, sz, &vqx, &vqy, &vqz);
    vx += vqx; vy += vqy; vz += vqz;
    vel[i * 3 + 0] = vx; vel[i * 3 + 1] = vy; vel[i * 3 + 2] = vz;
    // x_new = x + v_plus * dt
    pos[i * 3 + 0] += vx * dt;
    pos[i * 3 + 1] += vy * dt;
    pos[i * 3 + 2] += vz * dt;
}

// Trilinear interpolation: B at (x,y,z) from grid (nx,ny,nz), origin (ox,oy,oz), spacing (dx,dy,dz)
__device__ void sample_B_grid(const float* Bgrid, int nx, int ny, int nz,
                              float ox, float oy, float oz, float dx, float dy, float dz,
                              float x, float y, float z, float* Bx, float* By, float* Bz) {
    float fx = (x - ox) / dx;
    float fy = (y - oy) / dy;
    float fz = (z - oz) / dz;
    int i0 = (int)floorf(fx); int j0 = (int)floorf(fy); int k0 = (int)floorf(fz);
    if (i0 < 0) i0 = 0; if (i0 > nx - 2) i0 = nx - 2;
    if (j0 < 0) j0 = 0; if (j0 > ny - 2) j0 = ny - 2;
    if (k0 < 0) k0 = 0; if (k0 > nz - 2) k0 = nz - 2;
    int i1 = i0 + 1, j1 = j0 + 1, k1 = k0 + 1;
    float wx = fx - i0, wy = fy - j0, wz = fz - k0;
    auto idx = [nx, ny](int i, int j, int k) { return ((size_t)k * ny + j) * nx + i; };
    auto get = [Bgrid, idx, nx, ny, nz](int i, int j, int k, int c) {
        return Bgrid[(idx(i, j, k) * 3) + c];
    };
    float b000_x = get(i0, j0, k0, 0), b000_y = get(i0, j0, k0, 1), b000_z = get(i0, j0, k0, 2);
    float b100_x = get(i1, j0, k0, 0), b100_y = get(i1, j0, k0, 1), b100_z = get(i1, j0, k0, 2);
    float b010_x = get(i0, j1, k0, 0), b010_y = get(i0, j1, k0, 1), b010_z = get(i0, j1, k0, 2);
    float b110_x = get(i1, j1, k0, 0), b110_y = get(i1, j1, k0, 1), b110_z = get(i1, j1, k0, 2);
    float b001_x = get(i0, j0, k1, 0), b001_y = get(i0, j0, k1, 1), b001_z = get(i0, j0, k1, 2);
    float b101_x = get(i1, j0, k1, 0), b101_y = get(i1, j0, k1, 1), b101_z = get(i1, j0, k1, 2);
    float b011_x = get(i0, j1, k1, 0), b011_y = get(i0, j1, k1, 1), b011_z = get(i0, j1, k1, 2);
    float b111_x = get(i1, j1, k1, 0), b111_y = get(i1, j1, k1, 1), b111_z = get(i1, j1, k1, 2);
    float omwz = 1.f - wz;
    float c00_x = (1.f - wx) * b000_x + wx * b100_x, c00_y = (1.f - wx) * b000_y + wx * b100_y, c00_z = (1.f - wx) * b000_z + wx * b100_z;
    float c10_x = (1.f - wx) * b010_x + wx * b110_x, c10_y = (1.f - wx) * b010_y + wx * b110_y, c10_z = (1.f - wx) * b010_z + wx * b110_z;
    float c01_x = (1.f - wx) * b001_x + wx * b101_x, c01_y = (1.f - wx) * b001_y + wx * b101_y, c01_z = (1.f - wx) * b001_z + wx * b101_z;
    float c11_x = (1.f - wx) * b011_x + wx * b111_x, c11_y = (1.f - wx) * b011_y + wx * b111_y, c11_z = (1.f - wx) * b011_z + wx * b111_z;
    float c0_x = (1.f - wy) * c00_x + wy * c10_x, c0_y = (1.f - wy) * c00_y + wy * c10_y, c0_z = (1.f - wy) * c00_z + wy * c10_z;
    float c1_x = (1.f - wy) * c01_x + wy * c11_x, c1_y = (1.f - wy) * c01_y + wy * c11_y, c1_z = (1.f - wy) * c01_z + wy * c11_z;
    *Bx = omwz * c0_x + wz * c1_x;
    *By = omwz * c0_y + wz * c1_y;
    *Bz = omwz * c0_z + wz * c1_z;
}

__global__ void integrate_kernel_grid(float* pos, float* vel, const float* q, const float* m,
                                      const float* Bgrid, int nx, int ny, int nz,
                                      float ox, float oy, float oz, float dx, float dy, float dz,
                                      float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float Bx, By, Bz;
    sample_B_grid(Bgrid, nx, ny, nz, ox, oy, oz, dx, dy, dz,
                  pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2], &Bx, &By, &Bz);
    float qm = q[i] / m[i];
    float half_dt = dt * 0.5f;
    float tx = qm * Bx * half_dt, ty = qm * By * half_dt, tz = qm * Bz * half_dt;
    float vx = vel[i * 3 + 0], vy = vel[i * 3 + 1], vz = vel[i * 3 + 2];
    float vpx, vpy, vpz;
    cross(vx, vy, vz, tx, ty, tz, &vpx, &vpy, &vpz);
    vpx += vx; vpy += vy; vpz += vz;
    float t2 = tx*tx + ty*ty + tz*tz;
    float sx = 2.f * tx / (1.f + t2), sy = 2.f * ty / (1.f + t2), sz = 2.f * tz / (1.f + t2);
    float vqx, vqy, vqz;
    cross(vpx, vpy, vpz, sx, sy, sz, &vqx, &vqy, &vqz);
    vx += vqx; vy += vqy; vz += vqz;
    vel[i * 3 + 0] = vx; vel[i * 3 + 1] = vy; vel[i * 3 + 2] = vz;
    pos[i * 3 + 0] += vx * dt; pos[i * 3 + 1] += vy * dt; pos[i * 3 + 2] += vz * dt;
}

__global__ void record_kernel(const float* pos, float* traj, int n, int record_index, int rr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int base = (i * rr + record_index) * 3;
    traj[base + 0] = pos[i * 3 + 0];
    traj[base + 1] = pos[i * 3 + 1];
    traj[base + 2] = pos[i * 3 + 2];
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    const char* particles_path = (argc > 1) ? argv[1] : "data/particles.txt";
    const char* field_path     = (argc > 2) ? argv[2] : "data/field.txt";
    const char* params_path    = (argc > 3) ? argv[3] : "data/params.txt";
    const char* output_path   = (argc > 4) ? argv[4] : "out.bin";

    if (!load_particles(particles_path)) return 1;
    if (!load_field(field_path)) return 1;
    if (!load_params(params_path)) return 1;

    int num_steps = params.num_steps;
    int record_interval = params.record_interval;
    int rr = num_steps / record_interval;
    if (rr < 1) rr = 1;

    printf("Particles: %d, steps: %d, record_interval: %d, record points: %d\n",
           N, num_steps, record_interval, rr);

    // Device alloc
    float *d_pos = nullptr, *d_vel = nullptr, *d_q = nullptr, *d_m = nullptr, *d_traj = nullptr;
    float* d_Bgrid = nullptr;
    cudaMalloc(&d_pos, N * 3 * sizeof(float));
    cudaMalloc(&d_vel, N * 3 * sizeof(float));
    cudaMalloc(&d_q,  N * sizeof(float));
    cudaMalloc(&d_m,  N * sizeof(float));
    cudaMalloc(&d_traj, (size_t)N * rr * 3 * sizeof(float));
    if (use_B_grid && !h_Bgrid.empty()) {
        cudaMalloc(&d_Bgrid, h_Bgrid.size() * sizeof(float));
        cudaMemcpy(d_Bgrid, h_Bgrid.data(), h_Bgrid.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_pos, h_pos.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q,   h_q.data(),  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m,   h_m.data(),  N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;
    int record_count = 0;

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    for (int step = 0; step < num_steps; step++) {
        if (step % record_interval == 0) {
            record_kernel<<<grid, block>>>(d_pos, d_traj, N, record_count, rr);
            if (record_count == 0) {
                cudaEventRecord(start_ev);  // 第一步 record 完成后，即第一步积分 kernel 开始前
            }
            if (step == (rr - 1) * record_interval) {
                cudaEventRecord(stop_ev);  // 最后一个 record 结束
            }
            record_count++;
        }
        if (use_B_grid && d_Bgrid) {
            integrate_kernel_grid<<<grid, block>>>(d_pos, d_vel, d_q, d_m, d_Bgrid,
                g_nx, g_ny, g_nz, g_ox, g_oy, g_oz, g_dx, g_dy, g_dz,
                (float)params.dt, N);
        } else {
            integrate_kernel<<<grid, block>>>(d_pos, d_vel, d_q, d_m,
                h_B[0], h_B[1], h_B[2], (float)params.dt, N);
        }
    }
    cudaEventSynchronize(stop_ev);
    float gpu_ms = 0.f;
    cudaEventElapsedTime(&gpu_ms, start_ev, stop_ev);
    cudaEventDestroy(stop_ev);
    cudaEventDestroy(start_ev);

    double sim_time_sec = gpu_ms * 1e-3;
    // 计时区间为「第一步积分 kernel 开始 → 最后一个 record 结束」，区间内积分步数为 (rr-1)*record_interval
    int steps_timed = (rr >= 1) ? ((rr - 1) * record_interval) : 0;
    double particle_steps_per_sec = (steps_timed > 0 && sim_time_sec > 0)
        ? (N * (double)steps_timed) / sim_time_sec
        : (N * (double)num_steps) / sim_time_sec;
    size_t trajectory_bytes = sizeof(int) * 2 + (size_t)N * rr * 3 * sizeof(float);
    printf("Simulation GPU time: %.4f s\n", sim_time_sec);
    printf("Particle-step rate: %.2e (particle·steps/s)\n", particle_steps_per_sec);
    printf("Trajectory data size: %zu bytes (%.2f MB)\n",
           trajectory_bytes, trajectory_bytes / (1024.0 * 1024.0));

    // D2H and write binary
    std::vector<float> h_traj(N * rr * 3);
    cudaMemcpy(h_traj.data(), d_traj, h_traj.size() * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* out = fopen(output_path, "wb");
    if (out) {
        int pp = N, rr_out = rr;
        fwrite(&pp, sizeof(int), 1, out);
        fwrite(&rr_out, sizeof(int), 1, out);
        fwrite(h_traj.data(), sizeof(float), h_traj.size(), out);
        fclose(out);
        printf("Wrote %s (PP=%d RR=%d)\n", output_path, pp, rr_out);
    }

    if (d_Bgrid) cudaFree(d_Bgrid);
    cudaFree(d_traj); cudaFree(d_m); cudaFree(d_q); cudaFree(d_vel); cudaFree(d_pos);
    return 0;
}
