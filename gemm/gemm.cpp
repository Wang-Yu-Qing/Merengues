#include <iostream>
#include <stdint.h>
#include <time.h>
#include <random>
#include <cassert>
#include <chrono>

#include <xmmintrin.h>

// N should be big enough (ensure all matrix cannot fit into cache)
#define N 1024
#define BLOCK_SIZE 128

float A[N][N], B[N][N], C[N][N];

void initMat() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() / float(1e5);
      B[i][j] = rand() / float(1e5);
    }
  }
}

void transMatToM128(float (&src)[N][N], __m128 (&dst)[N][N / 4]) {
  assert(N % 4 == 0);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j += 4) {
      dst[i][j / 4] = _mm_set_ps(A[i][j], A[i][j + 1], A[i][j + 2], A[i][j + 3]);
      printf("%d, %d\n", i, j);
    }
  }
}

// for linux
//uint64_t nanos() {
//  struct timespec start;
//  clock_gettime(CLOCK_MONOTONIC, &start);
//  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
//}

// for windows


void printMat(float (&M)[N][N]) {
  printf("[\n");
  for (int i = 0; i < N; i++) {
    printf(" [");
    for (int j = 0; j < N; j++) {
      j == N - 1 ? printf("%f],\n", M[i][j]) : printf("%f, ", M[i][j]);
    }
  }
  printf("]\n");
}

// 0.545400 GFLOPS
void naive() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

// 0.926365 GFLOPS, 在c++中，array是按行存储的，因此按行方向遍历将充分利用缓存, https://www.bilibili.com/video/BV1cU4y1C7xY/?spm_id_from=333.999.0.0&vd_source=1edd1b65ec7bcde5814606bd07bca876
void trans() {
  // TODO: real transpose
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[j][k];
      }
    }
  }
}

/* 
  0.956145 GFLOPS
  通过block来限制进入缓存的矩阵A和B的数据大小（以免矩阵尺寸大时最先遍历的行被移出缓存），并在转移block前，充分地利用好已缓存的block，计算当前block的结果
  如果一行大到缓存吃不下，那么block不会比trans更好，否则block更好
  因为只有在能吃下block的情况下，缓存的数据才多于trans
*/
void blockTrans() {
  for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
      // 两方block的笛卡尔积全部遍历完，才转移到下一个block，在遍历期间充分利用了
      for (int i = bi; i < bi + BLOCK_SIZE; i++) {
        for (int j = bj; j < bj + BLOCK_SIZE; j++) {
          for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[j][k];
          }
        }
      }
    }
  }
}

int main() {
  printf("hello\n");
  initMat();
  __m128 _A[N][N / 4];
  transMatToM128(A, _A);

  printf("value is: %f\n", _mm_cvtss_f32(_A[0][0]));

  //uint64_t tic = nanos();
  auto tic = std::chrono::high_resolution_clock::now();

  naive();
  //trans();
  //blockTrans();

  //uint64_t toc = nanos();
  auto toc = std::chrono::high_resolution_clock::now();
  double gap = std::chrono::duration<double, std::milli>(tic - toc).count();

  long FLOP = 1LL * N * N * N * 2 * 1e-9;
  std::cout << N << "," << FLOP << std::endl;
  //printf("%f GFLOPS\n", FLOP / ((toc - tic) * 1e-9));
  printf("%f GFLOPS\n", FLOP / (gap * 1e-3));

  //printMat(C);

  return 0;
}