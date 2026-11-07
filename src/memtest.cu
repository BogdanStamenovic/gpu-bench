// VRAM integrity test. Allocates as much device memory as it can (minus a
// reserve), then runs pattern tests over all of it and counts mismatched words.
// Any nonzero error count means the VRAM (or memory controller) is faulty.
//
// Usage: memtest [--reserve-mib N] [--passes N] [--self-test]
// --self-test flips one bit after the first write: the run must then report exactly 1 error.
// Prints one JSON object on stdout. Exit 0 = no errors, 1 = errors, 2 = setup failure,
// 3 = interrupted with no errors.
// SIGTERM/SIGINT stop it between kernels (the watchdog uses this to yield to calls).

#include <cuda_runtime.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

#define CK(x) do { cudaError_t ck_e_ = (x); if (ck_e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(ck_e_), __FILE__, __LINE__); exit(2);} } while (0)

__device__ __forceinline__ uint32_t mix(uint64_t i, uint32_t seed) {
  // splitmix-style hash of the word index: a "random" pattern that is reproducible on read-back
  uint64_t z = i * 0x9E3779B97F4A7C15ull + seed;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return (uint32_t)(z ^ (z >> 31));
}

// mode 0: constant pattern; 1: address-in-address; 2: hashed random
__device__ __forceinline__ uint32_t expect(int mode, uint64_t gi, uint32_t pat) {
  if (mode == 0) return pat;
  if (mode == 1) return (uint32_t)gi ^ pat;
  return mix(gi, pat);
}

__global__ void k_write(uint32_t* p, size_t n, uint64_t base, int mode, uint32_t pat) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = expect(mode, base + i, pat);
}

__global__ void k_check(const uint32_t* p, size_t n, uint64_t base, int mode, uint32_t pat,
                        unsigned long long* errs, unsigned long long* bits) {
  unsigned long long e = 0, b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t d = p[i] ^ expect(mode, base + i, pat);
    if (d) { e++; b += __popc(d); }
  }
  if (e) { atomicAdd(errs, e); atomicAdd(bits, b); }
}

// moving inversion: verify value, write its complement, in one pass
__global__ void k_check_invert(uint32_t* p, size_t n, uint64_t base, int mode, uint32_t pat,
                               unsigned long long* errs, unsigned long long* bits) {
  unsigned long long e = 0, b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t want = expect(mode, base + i, pat);
    uint32_t d = p[i] ^ want;
    if (d) { e++; b += __popc(d); }
    p[i] = ~want;
  }
  if (e) { atomicAdd(errs, e); atomicAdd(bits, b); }
}

__global__ void k_check_inverted(const uint32_t* p, size_t n, uint64_t base, int mode, uint32_t pat,
                                 unsigned long long* errs, unsigned long long* bits) {
  unsigned long long e = 0, b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t d = p[i] ^ ~expect(mode, base + i, pat);
    if (d) { e++; b += __popc(d); }
  }
  if (e) { atomicAdd(errs, e); atomicAdd(bits, b); }
}

struct Test { const char* name; int mode; uint32_t pat; bool invert; };

int main(int argc, char** argv) {
  size_t reserve_mib = 400;
  int passes = 2;
  bool self_test = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--reserve-mib") && i + 1 < argc) reserve_mib = atol(argv[++i]);
    else if (!strcmp(argv[i], "--passes") && i + 1 < argc) passes = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--self-test")) self_test = true;
    else { fprintf(stderr, "usage: memtest [--reserve-mib N] [--passes N] [--self-test]\n"); return 2; }
  }
  signal(SIGTERM, on_sig); signal(SIGINT, on_sig);

  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  size_t free_b, total_b; CK(cudaMemGetInfo(&free_b, &total_b));

  // Allocate in 256 MiB chunks until we hit the reserve: one huge allocation
  // can fail on fragmentation even when the memory is there.
  const size_t chunk = 256ull << 20;
  std::vector<uint32_t*> bufs;
  size_t allocated = 0;
  while (true) {
    CK(cudaMemGetInfo(&free_b, &total_b));
    if (free_b < (reserve_mib << 20) + chunk) break;
    uint32_t* p = nullptr;
    if (cudaMalloc(&p, chunk) != cudaSuccess) { cudaGetLastError(); break; }
    bufs.push_back(p); allocated += chunk;
  }
  if (bufs.empty()) { fprintf(stderr, "could not allocate any VRAM\n"); return 2; }
  fprintf(stderr, "memtest: %s, testing %zu MiB of %zu MiB in %zu chunks, %d passes\n",
          prop.name, allocated >> 20, total_b >> 20, bufs.size(), passes);

  unsigned long long *d_err, *d_bits;
  CK(cudaMalloc(&d_err, 8)); CK(cudaMalloc(&d_bits, 8));

  const Test tests[] = {
    {"zeros", 0, 0x00000000u, false}, {"ones", 0, 0xFFFFFFFFu, false},
    {"checker_a", 0, 0xAAAAAAAAu, false}, {"checker_5", 0, 0x55555555u, false},
    {"addr_in_addr", 1, 0u, false}, {"addr_in_addr_inv", 1, 0xFFFFFFFFu, false},
    {"random_moving_inv", 2, 0x1234567u, true}, {"walking_ones_moving_inv", 0, 0x01010101u, true},
    {"random2", 2, 0xC0FFEEu, false},
  };
  const size_t words = chunk / 4;
  const int grid = prop.multiProcessorCount * 8, block = 256;

  std::string results = "[";
  unsigned long long total_err = 0, total_bits = 0;
  double gb_moved = 0;
  auto t0 = std::chrono::steady_clock::now();
  bool interrupted = false;

  for (int pass = 0; pass < passes && !interrupted; pass++) {
    for (const Test& t : tests) {
      if (g_stop) { interrupted = true; break; }
      CK(cudaMemset(d_err, 0, 8)); CK(cudaMemset(d_bits, 0, 8));
      auto ts = std::chrono::steady_clock::now();
      for (size_t c = 0; c < bufs.size(); c++)
        k_write<<<grid, block>>>(bufs[c], words, (uint64_t)c * words, t.mode, t.pat);
      CK(cudaDeviceSynchronize());
      if (self_test && pass == 0 && &t == &tests[0]) {
        // flip one bit in the last chunk to prove the checker sees it
        uint32_t v; uint32_t* where = bufs.back() + 12345;
        CK(cudaMemcpy(&v, where, 4, cudaMemcpyDeviceToHost)); v ^= 0x10u;
        CK(cudaMemcpy(where, &v, 4, cudaMemcpyHostToDevice));
      }
      for (size_t c = 0; c < bufs.size(); c++) {
        if (t.invert) k_check_invert<<<grid, block>>>(bufs[c], words, (uint64_t)c * words, t.mode, t.pat, d_err, d_bits);
        else k_check<<<grid, block>>>(bufs[c], words, (uint64_t)c * words, t.mode, t.pat, d_err, d_bits);
      }
      if (t.invert) {
        CK(cudaDeviceSynchronize());
        for (size_t c = 0; c < bufs.size(); c++)
          k_check_inverted<<<grid, block>>>(bufs[c], words, (uint64_t)c * words, t.mode, t.pat, d_err, d_bits);
      }
      CK(cudaDeviceSynchronize());
      double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
      unsigned long long e, b;
      CK(cudaMemcpy(&e, d_err, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&b, d_bits, 8, cudaMemcpyDeviceToHost));
      total_err += e; total_bits += b;
      double moved = (double)allocated * (t.invert ? 4 : 2) / 1e9;  // write + read (+ write + read)
      gb_moved += moved;
      fprintf(stderr, "  pass %d %-24s errors=%llu  %.1f GB/s\n", pass + 1, t.name, e, moved / s);
      char buf[256];
      snprintf(buf, sizeof buf, "%s{\"pass\":%d,\"test\":\"%s\",\"errors\":%llu,\"bit_flips\":%llu,\"seconds\":%.3f}",
               results.size() > 1 ? "," : "", pass + 1, t.name, e, b, s);
      results += buf;
    }
  }
  results += "]";
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  printf("{\"device\":\"%s\",\"tested_mib\":%zu,\"total_mib\":%zu,\"passes_requested\":%d,"
         "\"interrupted\":%s,\"errors\":%llu,\"bit_flips\":%llu,\"seconds\":%.2f,"
         "\"effective_gbps\":%.1f,\"tests\":%s}\n",
         prop.name, allocated >> 20, total_b >> 20, passes, interrupted ? "true" : "false",
         total_err, total_bits, secs, gb_moved / secs, results.c_str());
  return total_err ? 1 : (interrupted ? 3 : 0);
}
