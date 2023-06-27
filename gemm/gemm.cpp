#include <iostream>
#include <stdint.h>
#include <time.h>
#include <random>

#define N 10

const int K = 10;
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

void printMat() {
  printf("[\n");
  for (int i = 0; i < N; i++) {
    printf(" [");
    for (int j = 0; j < N; j++) {
      j == N - 1 ? printf("%f],\n", C[i][j]) : printf("%f, ", C[i][j]);
    }
  }
  printf("]\n");
}

void naive() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main() {
  initMat();

  uint64_t tic = nanos();
  for (int i = 0; i < K; i++) {
    naive();
  }
  uint64_t toc = nanos();

  printf("%f GFLOPS\n", (K * N * N * N * 2 * 1e-9) / ((toc - tic) * 1e-9));

  printMat();

  return 0;
}