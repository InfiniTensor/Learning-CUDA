#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <parallel/algorithm>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define MATCH_SCORE 2
#define MISMATCH_SCORE -1
#define GAP_PENALTY -1
#define MAX_READ_LEN 256
#define KMER_SIZE 16
constexpr int LOOKUP_PREFIX_BITS = 22;
constexpr uint32_t LOOKUP_BUCKETS = (1u << LOOKUP_PREFIX_BITS);
constexpr int LOOKUP_SHIFT = 32 - LOOKUP_PREFIX_BITS;

struct Read {
    std::string name;
    std::string seq;
};

// Modified to hold read kmers instead of reference kmers
struct KmerPos {
    uint32_t kmer;
    int read_id;
    int offset;
    bool operator<(const KmerPos &other) const { return kmer < other.kmer; }
};

void load_reference(const std::string &filename, std::vector<std::string> &names, std::string &concat_seq,
                    std::vector<size_t> &offsets) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        exit(1);

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        exit(1);
    size_t size = sb.st_size;

    const char *mapped_ptr = static_cast<const char *>(mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (mapped_ptr == MAP_FAILED)
        exit(1);
    close(fd);

    concat_seq.reserve(size);

    const char *ptr = mapped_ptr;
    const char *end = ptr + size;
    bool is_fastq = (ptr < end && *ptr == '@');

    while (ptr < end) {
        if (is_fastq) {
            const char *name_end = (const char *)memchr(ptr, '\n', end - ptr);
            if (!name_end)
                name_end = end;
            std::string name(ptr + 1, name_end - (ptr + 1));
            if (!name.empty() && name.back() == '\r')
                name.pop_back();
            ptr = name_end + (name_end < end ? 1 : 0);

            if (ptr >= end)
                break;
            const char *seq_end = (const char *)memchr(ptr, '\n', end - ptr);
            if (!seq_end)
                seq_end = end;
            std::string seq(ptr, seq_end - ptr);
            if (!seq.empty() && seq.back() == '\r')
                seq.pop_back();
            ptr = seq_end + (seq_end < end ? 1 : 0);

            names.push_back(name);
            offsets.push_back(concat_seq.size());
            concat_seq += seq;

            ptr = (const char *)memchr(ptr, '\n', end - ptr);
            if (ptr)
                ptr++;
            else
                break;
            ptr = (const char *)memchr(ptr, '\n', end - ptr);
            if (ptr)
                ptr++;
            else
                break;
        } else {
            if (*ptr == '>') {
                const char *name_end = (const char *)memchr(ptr, '\n', end - ptr);
                if (!name_end)
                    name_end = end;
                std::string name(ptr + 1, name_end - (ptr + 1));
                if (!name.empty() && name.back() == '\r')
                    name.pop_back();
                names.push_back(name);
                offsets.push_back(concat_seq.size());
                ptr = name_end + (name_end < end ? 1 : 0);
            } else {
                const char *seq_end = (const char *)memchr(ptr, '\n', end - ptr);
                if (!seq_end)
                    seq_end = end;
                size_t seq_len = seq_end - ptr;
                if (seq_len > 0 && *(seq_end - 1) == '\r')
                    seq_len--;
                concat_seq.append(ptr, seq_len);
                ptr = seq_end + (seq_end < end ? 1 : 0);
            }
        }
    }
    munmap((void *)mapped_ptr, size);
}

void load_reads(const std::string &filename, std::vector<Read> &reads) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        exit(1);

    struct stat sb;
    if (fstat(fd, &sb) == -1)
        exit(1);
    size_t size = sb.st_size;

    const char *mapped_ptr = static_cast<const char *>(mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (mapped_ptr == MAP_FAILED)
        exit(1);
    close(fd);

    reads.reserve(size / 150);

    const char *ptr = mapped_ptr;
    const char *end = ptr + size;

    while (ptr < end) {
        if (*ptr != '@') {
            ptr = (const char *)memchr(ptr, '\n', end - ptr);
            if (ptr)
                ptr++;
            else
                break;
            continue;
        }

        Read r;
        const char *name_end = (const char *)memchr(ptr, '\n', end - ptr);
        if (!name_end)
            name_end = end;
        r.name.assign(ptr + 1, name_end - (ptr + 1));
        if (!r.name.empty() && r.name.back() == '\r')
            r.name.pop_back();
        ptr = name_end + (name_end < end ? 1 : 0);

        if (ptr >= end)
            break;

        const char *seq_end = (const char *)memchr(ptr, '\n', end - ptr);
        if (!seq_end)
            seq_end = end;
        r.seq.assign(ptr, seq_end - ptr);
        if (!r.seq.empty() && r.seq.back() == '\r')
            r.seq.pop_back();
        ptr = seq_end + (seq_end < end ? 1 : 0);

        ptr = (const char *)memchr(ptr, '\n', end - ptr);
        if (ptr)
            ptr++;
        else
            break;
        ptr = (const char *)memchr(ptr, '\n', end - ptr);
        if (ptr)
            ptr++;
        else
            break;

        reads.push_back(std::move(r));
    }
    munmap((void *)mapped_ptr, size);
}

inline uint8_t char2val_cpu(char c) { return (c >> 1) & 0x03; }

inline uint32_t pack_16mer(const char *seq) {
    uint32_t packed = 0;
    for (int i = 0; i < 16; ++i) {
        uint8_t val = char2val_cpu(seq[i]);
        packed |= (val << (30 - 2 * i));
    }
    return packed;
}

void encode_sequence_cpu(const std::string &seq, uint32_t *buffer) {
    size_t length = seq.length();
    size_t encoded_size = (length + 15) / 16;

    // 小任务串行，避免 OMP 开销
    if (encoded_size < 4096) {
        for (size_t i = 0; i < encoded_size; ++i) {
            uint32_t packed = 0;
            size_t start = i * 16;
            for (int j = 0; j < 16; ++j) {
                if (start + j < length) {
                    uint8_t val = char2val_cpu(seq[start + j]);
                    packed |= (uint32_t)(val << (30 - 2 * j));
                }
            }
            buffer[i] = packed;
        }
        return;
    }

#pragma omp parallel for
    for (size_t i = 0; i < encoded_size; ++i) {
        uint32_t packed = 0;
        size_t start = i * 16;
        for (int j = 0; j < 16; ++j) {
            if (start + j < length) {
                uint8_t val = char2val_cpu(seq[start + j]);
                packed |= (uint32_t)(val << (30 - 2 * j));
            }
        }
        buffer[i] = packed;
    }
}

__device__ inline uint8_t get_base(const uint32_t *packed_array, size_t index) {
    size_t array_idx = index / 16;
    int bit_pos = 30 - 2 * (index % 16);
    return (packed_array[array_idx] >> bit_pos) & 0x03;
}

extern "C" __global__ void sw_extend_wavefront_kernel(const uint32_t *ref_packed, size_t ref_len,
                                                      const uint32_t *reads_packed, const int *read_lengths,
                                                      const int64_t *candidate_positions,
                                                      const int *candidate_read_indices, int total_candidates,
                                                      int *out_scores, int64_t *out_best_pos) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int local_warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (global_warp_id >= total_candidates)
        return;

    extern __shared__ short shared_buf[];

    int warps_per_block = blockDim.x / 32;
    short *H_row = &shared_buf[local_warp_id * MAX_READ_LEN];
    short *Start_row = &shared_buf[local_warp_id * MAX_READ_LEN + (MAX_READ_LEN * warps_per_block)];

    int64_t ref_start = candidate_positions[global_warp_id];
    int read_idx = candidate_read_indices[global_warp_id];
    int read_len = read_lengths[read_idx];

    const uint32_t *my_read_packed = reads_packed + read_idx * (MAX_READ_LEN / 16);

    int ref_region_len = read_len + 100;
    if (ref_start + ref_region_len > ref_len) {
        ref_region_len = ref_len - ref_start;
    }

    int total_chunks = (read_len + 31) / 32;

    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        int j = chunk * 32 + lane_id;
        if (j < read_len) {
            H_row[j] = 0;
            Start_row[j] = -1;
        }
    }

    __syncwarp();

    short max_score = 0;
    short best_ref_offset = 0;

    short h_diag_reg[8] = {0};
    short start_diag_reg[8] = {0};

    short next_h_reg[8] = {0};
    short next_start_reg[8] = {-1};

    int total_steps = ref_region_len + read_len - 1;

    for (int step = 0; step < total_steps; ++step) {

        for (int chunk = 0; chunk < total_chunks; ++chunk) {
            int j = chunk * 32 + lane_id;
            int i = step - j;

            if (i >= 0 && i < ref_region_len && j < read_len) {
                int64_t current_ref_pos = ref_start + i;
                uint8_t ref_base = get_base(ref_packed, current_ref_pos);
                uint8_t read_base = get_base(my_read_packed, j);

                short match = (ref_base == read_base) ? MATCH_SCORE : MISMATCH_SCORE;

                short h_up = H_row[j];
                short start_up = Start_row[j];

                short h_left = (j > 0) ? H_row[j - 1] : 0;
                short start_left = (j > 0) ? Start_row[j - 1] : -1;

                short score_diag = h_diag_reg[chunk] + match;
                short score_up = h_up + GAP_PENALTY;
                short score_left = h_left + GAP_PENALTY;

                short score = 0;
                short start = i;

                if (score_diag > 0 && score_diag >= score_up && score_diag >= score_left) {
                    score = score_diag;
                    start = (h_diag_reg[chunk] == 0) ? i : start_diag_reg[chunk];
                } else if (score_up > 0 && score_up >= score_left) {
                    score = score_up;
                    start = (h_up == 0) ? i : start_up;
                } else if (score_left > 0) {
                    score = score_left;
                    start = (h_left == 0) ? i : start_left;
                }

                // 【修复核心】将当前步的左侧值，作为下一步的对角线值缓存！
                h_diag_reg[chunk] = h_left;
                start_diag_reg[chunk] = start_left;

                next_h_reg[chunk] = score;
                next_start_reg[chunk] = start;

                if (score > max_score) {
                    max_score = score;
                    best_ref_offset = start;
                }
            }
        }

        __syncwarp();

        for (int chunk = 0; chunk < total_chunks; ++chunk) {
            int j = chunk * 32 + lane_id;
            int i = step - j;
            if (i >= 0 && i < ref_region_len && j < read_len) {
                H_row[j] = next_h_reg[chunk];
                Start_row[j] = next_start_reg[chunk];
            }
        }

        __syncwarp();
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        short shfl_score = __shfl_down_sync(0xffffffff, max_score, offset);
        short shfl_offset = __shfl_down_sync(0xffffffff, best_ref_offset, offset);
        if (shfl_score > max_score) {
            max_score = shfl_score;
            best_ref_offset = shfl_offset;
        }
    }

    if (lane_id == 0) {
        out_scores[global_warp_id] = max_score;
        out_best_pos[global_warp_id] = ref_start + best_ref_offset;
    }
}

int main() {
        cudaFree(0);

    std::string ref_file = "reference.fasta";
    std::string reads_file = "reads.fastq";
    std::ofstream out_file("results.txt");

    std::vector<std::string> ref_names;
    std::vector<size_t> ref_offsets;
    std::string concat_ref;

        load_reference(ref_file, ref_names, concat_ref, ref_offsets);

        concat_ref.append(100, 'N');

    std::vector<Read> reads;
        load_reads(reads_file, reads);

    int num_reads = reads.size();

    size_t ref_encoded_size = (concat_ref.length() + 15) / 16;
    uint32_t *h_ref_packed = new uint32_t[ref_encoded_size];

        encode_sequence_cpu(concat_ref, h_ref_packed);

    uint32_t *d_ref_packed;
        cudaMalloc(&d_ref_packed, ref_encoded_size * sizeof(uint32_t));
    

        cudaMemcpy(d_ref_packed, h_ref_packed, ref_encoded_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    

    std::vector<uint32_t> h_reads_packed(num_reads * (MAX_READ_LEN / 16), 0);
    std::vector<int> h_read_lengths(num_reads, 0);
    std::vector<KmerPos> kmer_list;
    kmer_list.reserve(num_reads * 50);

    for (int i = 0; i < num_reads; ++i) {
        int rlen = std::min((int)reads[i].seq.length(), MAX_READ_LEN);
        h_read_lengths[i] = rlen;
        encode_sequence_cpu(reads[i].seq, h_reads_packed.data() + i * (MAX_READ_LEN / 16));

        for (int offset = 0; offset <= rlen - KMER_SIZE; offset += 2) {
            kmer_list.push_back({pack_16mer(reads[i].seq.c_str() + offset), i, offset});
        }
    }
    __gnu_parallel::sort(kmer_list.begin(), kmer_list.end());

    // 22-bit prefix lookup
    std::vector<uint32_t> lookup(LOOKUP_BUCKETS + 1, (uint32_t)kmer_list.size());
    uint32_t prefix = 0;
    lookup[0] = 0;
    for (size_t i = 0; i < kmer_list.size(); ++i) {
        uint32_t p = kmer_list[i].kmer >> LOOKUP_SHIFT;
        while (prefix < p)
            lookup[++prefix] = (uint32_t)i;
    }
    while (prefix < LOOKUP_BUCKETS)
        lookup[++prefix] = (uint32_t)kmer_list.size();

    std::vector<std::vector<int64_t>> read_cands(num_reads);

    // rolling k-mer scan on reference
    if (concat_ref.length() >= (KMER_SIZE + 100)) {
        const int64_t max_pos = (int64_t)concat_ref.length() - KMER_SIZE - 100; // inclusive

#pragma omp parallel
        {
            std::vector<std::vector<int64_t>> local_cands(num_reads);

            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();

            const int64_t total = max_pos + 1; // pos in [0, max_pos]
            const int64_t chunk = (total + nth - 1) / nth;
            const int64_t begin = tid * chunk;
            const int64_t end = std::min<int64_t>(total, begin + chunk);

            if (begin < end) {
                uint32_t kmer = pack_16mer(concat_ref.data() + begin);

                for (int64_t pos = begin; pos < end; ++pos) {
                    uint32_t p = kmer >> LOOKUP_SHIFT;

                    for (uint32_t i = lookup[p]; i < lookup[p + 1]; ++i) {
                        if (kmer_list[i].kmer == kmer) {
                            int64_t start = pos - kmer_list[i].offset - 50;
                            if (start >= 0)
                                local_cands[kmer_list[i].read_id].push_back(start);
                        }
                    }

                    // roll to next 16-mer
                    if (pos + 1 < end) {
                        kmer = (kmer << 2) | (uint32_t)char2val_cpu(concat_ref[pos + KMER_SIZE]);
                    }
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < num_reads; ++i) {
                    read_cands[i].insert(read_cands[i].end(), local_cands[i].begin(), local_cands[i].end());
                }
            }
        }
    }

    std::vector<int64_t> h_candidate_pos;
    std::vector<int> h_candidate_read_idx;

    for (int i = 0; i < num_reads; ++i) {
        if (read_cands[i].empty())
            continue;
        std::sort(read_cands[i].begin(), read_cands[i].end());
        int64_t last_added = -1000;
        int count = 0;
        for (int64_t pos : read_cands[i]) {
            if (pos > last_added + 50) {
                h_candidate_pos.push_back(pos);
                h_candidate_read_idx.push_back(i);
                last_added = pos;
                if (++count >= 100)
                    break;
            }
        }
    }

    int total_candidates = h_candidate_pos.size();
    if (total_candidates == 0) {
        for (const auto &r : reads)
            out_file << r.name << " unknown_origin\n";
        return 0;
    }

    uint32_t *d_reads_packed;
    int *d_read_lengths, *d_candidate_read_idx, *d_out_scores;
    int64_t *d_candidate_pos, *d_out_best_pos;

    cudaMalloc(&d_reads_packed, h_reads_packed.size() * sizeof(uint32_t));
    cudaMalloc(&d_read_lengths, num_reads * sizeof(int));
    cudaMalloc(&d_candidate_pos, total_candidates * sizeof(int64_t));
    cudaMalloc(&d_candidate_read_idx, total_candidates * sizeof(int));
    cudaMalloc(&d_out_scores, total_candidates * sizeof(int));
    cudaMalloc(&d_out_best_pos, total_candidates * sizeof(int64_t));

    cudaMemcpy(d_reads_packed, h_reads_packed.data(), h_reads_packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_read_lengths, h_read_lengths.data(), num_reads * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate_pos, h_candidate_pos.data(), total_candidates * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate_read_idx, h_candidate_read_idx.data(), total_candidates * sizeof(int),
               cudaMemcpyHostToDevice);

    int blockSize = 128; // 4 Warps per Block
    // Total threads needed = total_candidates * 32
    int numBlocks = (total_candidates * 32 + blockSize - 1) / blockSize;

    int warps_per_block = blockSize / 32;
    // Shared memory: warps_per_block * (MAX_READ_LEN * 2 arrays) * sizeof(short)
    size_t sharedMemSize = warps_per_block * MAX_READ_LEN * 2 * sizeof(short);

    sw_extend_wavefront_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_ref_packed, concat_ref.length() - 100, d_reads_packed, d_read_lengths, d_candidate_pos, d_candidate_read_idx,
        total_candidates, d_out_scores, d_out_best_pos);

    cudaDeviceSynchronize();

    std::vector<int> h_out_scores(total_candidates);
    std::vector<int64_t> h_out_best_pos(total_candidates);
    cudaMemcpy(h_out_scores.data(), d_out_scores, total_candidates * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_best_pos.data(), d_out_best_pos, total_candidates * sizeof(int64_t), cudaMemcpyDeviceToHost);

    std::vector<int> best_score_per_read(num_reads, -1);
    std::vector<int64_t> best_pos_per_read(num_reads, -1);

    for (int i = 0; i < total_candidates; ++i) {
        int r_idx = h_candidate_read_idx[i];
        if (h_out_scores[i] > best_score_per_read[r_idx]) {
            best_score_per_read[r_idx] = h_out_scores[i];
            best_pos_per_read[r_idx] = h_out_best_pos[i];
        }
    }

    for (int i = 0; i < num_reads; ++i) {
        if (best_score_per_read[i] > h_read_lengths[i] * 0.5) {
            int64_t global_pos = best_pos_per_read[i];
            auto it = std::upper_bound(ref_offsets.begin(), ref_offsets.end(), global_pos);
            int ref_idx = std::distance(ref_offsets.begin(), it) - 1;
            int64_t local_pos = global_pos - ref_offsets[ref_idx];

            out_file << reads[i].name << " " << ref_names[ref_idx] << " " << local_pos << "\n";
        } else {
            out_file << reads[i].name << " unknown_origin\n";
        }
    }

    cudaFree(d_ref_packed);
    cudaFree(d_reads_packed);
    cudaFree(d_read_lengths);
    cudaFree(d_candidate_pos);
    cudaFree(d_candidate_read_idx);
    cudaFree(d_out_scores);
    cudaFree(d_out_best_pos);
    delete[] h_ref_packed;

    return 0;
}
