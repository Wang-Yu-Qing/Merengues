#include <iostream>
#include <stdint.h>
#include <time.h>
#include <random>

// N should be big enough (ensure all matrix cannot fit into cache)
#define N 1000
#define BLOCK_SIZE 100

float A[N][N], B[N][N], C[N][N];

void initMat() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() / float(1e5);
      B[i][j] = rand() / float(1e5);
    }
  }
}

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

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
  如果一行大到缓存吃不下，那么block不会比trans更好，否则block更好
  因为只有在能吃下block的情况下，缓存的数据才多于trans
*/
void blockTrans() {
  for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
    for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
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
  initMat();

  uint64_t tic = nanos();
  //naive();
  //trans();
  blockTrans();
  uint64_t toc = nanos();

  printf("%f GFLOPS\n", (N * N * N * 2 * 1e-9) / ((toc - tic) * 1e-9));

  //printMat(C);

  return 0;
}