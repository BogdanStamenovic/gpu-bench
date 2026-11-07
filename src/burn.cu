// Sustained compute burn with result verification (the gpu-burn idea).
// Repeats C = A*B with cuBLAS and compares each result against the first one.
// A healthy card produces the same result every time; mismatches mean the
// silicon, VRAM or power delivery is producing wrong answers under load.
//
// Usage: burn --mode fp32|fp16 --seconds S [--n N] [--sample-s S] [--self-test]
// --self-test perturbs one reference element: every check must then report exactly 1 mismatch.
//   fp32: SGEMM on CUDA cores. fp16: half inputs, fp32 accumulate, tensor cores.
// Prints one JSON object on stdout. Exit 0 = clean, 1 = mismatches,
// 2 = setup failure, 3 = interrupted (SIGTERM/SIGINT) with no mismatches.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

#define CK(x) do { cudaError_t ck_e_ = (x); if (ck_e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(ck_e_), __FILE__, __LINE__); exit(2);} } while (0)
#define CB(x) do { cublasStatus_t cb_s_ = (x); if (cb_s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)cb_s_, __FILE__, __LINE__); exit(2);} } while (0)

__device__ float rnd(uint64_t i, uint32_t seed) {
  uint64_t z = i * 0x9E3779B97F4A7C15ull + seed;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return ((uint32_t)(z >> 40) / 16777216.0f) * 2.0f - 1.0f;
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = rnd(i, seed);
}
__global__ void k_fill_f16(__half* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = __float2half(rnd(i, seed));
}
// Count elements that differ from the reference beyond a small tolerance.
// cuBLAS is deterministic for a fixed config on one GPU, so in practice a
// healthy card gives zero difference; the tolerance only absorbs harmless
// reordering if the library picks a different kernel mid-run.
__global__ void k_compare(const float* c, const float* ref, size_t n, unsigned long long* bad) {
  unsigned long long b = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float d = fabsf(c[i] - ref[i]);
    if (!(d <= 1e-3f * fabsf(ref[i]) + 1e-3f)) b++;   // also catches NaN
  }
  if (b) atomicAdd(bad, b);
}

int main(int argc, char** argv) {
  std::string mode = "fp32";
  double seconds = 60, sample_s = 10;
  int n = 8192;
  bool self_test = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--sample-s") && i + 1 < argc) sample_s = atof(argv[++i]);
    else if (!strcmp(argv[i], "--self-test")) self_test = true;
    else { fprintf(stderr, "usage: burn --mode fp32|fp16 --seconds S [--n N] [--sample-s S]\n"); return 2; }
  }
  if (mode != "fp32" && mode != "fp16") { fprintf(stderr, "mode must be fp32 or fp16\n"); return 2; }
  signal(SIGTERM, on_sig); signal(SIGINT, on_sig);

  CK(cudaSetDevice(0));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  cublasHandle_t h; CB(cublasCreate(&h));
  CB(cublasSetMathMode(h, mode == "fp16" ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH));  // fp32: no TF32 shortcut

  size_t nn = (size_t)n * n;
  size_t esz = mode == "fp32" ? 4 : 2;
  void *A, *B; float *C, *R;
  CK(cudaMalloc(&A, nn * esz)); CK(cudaMalloc(&B, nn * esz));
  CK(cudaMalloc(&C, nn * 4)); CK(cudaMalloc(&R, nn * 4));
  unsigned long long* d_bad; CK(cudaMalloc(&d_bad, 8));
  int grid = prop.multiProcessorCount * 8;
  if (mode == "fp32") { k_fill_f32<<<grid, 256>>>((float*)A, nn, 1); k_fill_f32<<<grid, 256>>>((float*)B, nn, 2); }
  else { k_fill_f16<<<grid, 256>>>((__half*)A, nn, 1); k_fill_f16<<<grid, 256>>>((__half*)B, nn, 2); }
  CK(cudaDeviceSynchronize());

  const float one = 1.0f, zero = 0.0f;
  auto gemm = [&](float* out) {
    if (mode == "fp32")
      CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, (float*)A, n, (float*)B, n, &zero, out, n));
    else
      CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, CUDA_R_16F, n, B, CUDA_R_16F, n,
                      &zero, out, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  gemm(R);                    // reference result
  CK(cudaDeviceSynchronize());
  if (self_test) {            // perturb one reference element: every check must then report 1 mismatch
    float v; CK(cudaMemcpy(&v, R + 777, 4, cudaMemcpyDeviceToHost)); v += 1.0f;
    CK(cudaMemcpy(R + 777, &v, 4, cudaMemcpyHostToDevice));
  }

  const double flop = 2.0 * n * (double)n * n;
  unsigned long long iters = 0, checks = 0, bad_total = 0, bad_checks = 0;
  std::string series = "[";
  auto t0 = std::chrono::steady_clock::now(), ts = t0;
  unsigned long long it_sample = 0;
  double min_tf = 1e30, max_tf = 0;
  while (!g_stop) {
    for (int k = 0; k < 4; k++) gemm(C);   // 4 GEMMs per check keeps the compare overhead small
    CK(cudaMemset(d_bad, 0, 8));
    k_compare<<<grid, 256>>>(C, R, nn, d_bad);
    unsigned long long bad;
    CK(cudaMemcpy(&bad, d_bad, 8, cudaMemcpyDeviceToHost));   // implicit sync
    iters += 4; it_sample += 4; checks++;
    if (bad) { bad_total += bad; bad_checks++; fprintf(stderr, "burn: MISMATCH %llu elements at check %llu\n", bad, checks); }
    auto now = std::chrono::steady_clock::now();
    double el = std::chrono::duration<double>(now - t0).count();
    double sl = std::chrono::duration<double>(now - ts).count();
    if (sl >= sample_s) {
      double tf = it_sample * flop / sl / 1e12;
      if (tf < min_tf) min_tf = tf;
      if (tf > max_tf) max_tf = tf;
      char b[96]; snprintf(b, sizeof b, "%s[%.1f,%.2f]", series.size() > 1 ? "," : "", el, tf);
      series += b;
      fprintf(stderr, "burn %s t=%5.0fs  %.2f TFLOPS  mismatches=%llu\n", mode.c_str(), el, tf, bad_total);
      ts = now; it_sample = 0;
    }
    if (el >= seconds) break;
  }
  series += "]";
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("{\"device\":\"%s\",\"mode\":\"%s\",\"n\":%d,\"seconds\":%.1f,\"interrupted\":%s,\"gemms\":%llu,"
         "\"checks\":%llu,\"mismatched_elements\":%llu,\"mismatched_checks\":%llu,"
         "\"tflops_avg\":%.2f,\"tflops_min_sample\":%.2f,\"tflops_max_sample\":%.2f,\"tflops_series\":%s}\n",
         prop.name, mode.c_str(), n, el, g_stop ? "true" : "false", iters, checks, bad_total, bad_checks,
         iters * flop / el / 1e12, min_tf > 1e29 ? 0.0 : min_tf, max_tf, series.c_str());
  return bad_total ? 1 : (g_stop ? 3 : 0);
}
