#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

/* INT_MAX < LONG_MAX < ULONG_MAX < LLONG_MAX < ULLONG_MAX*/
typedef unsigned vector_t;
typedef unsigned int histogram_t;
typedef int atomic_t;
#define MIL 1000
#define MILLON MIL*MIL
#define N 100*MIL
#define M 8       //Tamaño histograma
#define SIZE_BYTES sizeof(vector_t);
#define R 106     //Repeticiones

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void inicializar_histograma(histogram_t * h, size_t n){
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(thread_id >= n) return;
  h[thread_id] = 0;
}
__global__ void calcular_histograma(vector_t* v, histogram_t* h, 
    size_t n, size_t m){

  size_t id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= n) return;
  atomicAdd(&h[v[id]%m], 1);
}

float tiempo_kernel(histogram_t * h_device, vector_t * v_device, unsigned kernel);

int main(void){
  srand(time(NULL));

  size_t sizeV = N * sizeof(vector_t);
  vector_t * v = (vector_t *)malloc(sizeV);
  for (size_t i = 0; i < N; i++)
    v[i] = rand()%M;
  
  vector_t * v_device;
  CUDA_CHECK_RETURN(cudaMalloc(&v_device, sizeV)); 
  CUDA_CHECK_RETURN(cudaMemcpy(v_device, v, sizeV, cudaMemcpyHostToDevice));

  size_t sizeH = M * sizeof(histogram_t);
  histogram_t * h = (histogram_t *)malloc(sizeH);

  histogram_t * h_device;
  CUDA_CHECK_RETURN(cudaMalloc(&h_device, sizeH)); 

  printf("Llamando kernel con M = %i N = %i y P = %i\n", M, N, 1);

  float totalTime = tiempo_kernel(h_device, v_device, 1); 

  CUDA_CHECK_RETURN(cudaMemcpy(h, h_device, sizeH, cudaMemcpyDeviceToHost));

  printf("Tiempo transcurrido: %f ms\n", totalTime);
  printf("Histograma: ");
  size_t suma = 0;
  for (int i = 0; i < M; i++){
    printf("%u ", h[i]);
    suma += h[i];
  }
  printf("\n");
  printf("Elementos del vector: %i\n", N);
  printf("Suma de elementos : %lu\n", suma);

  free(v);
  free(h);
  CUDA_CHECK_RETURN(cudaFree(v_device));
  CUDA_CHECK_RETURN(cudaFree(h_device));
}

float tiempo_kernel(histogram_t * h_device, vector_t * v_device, unsigned kernel){
  size_t threadsPerBlock = 1024;
  size_t blocksPerGridM = ((unsigned)M + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksPerGridN = ((unsigned)N + threadsPerBlock - 1) / threadsPerBlock;
  
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  double totalTime = 0;


  for(int i = 0; i < R; i++){
    cudaEventRecord(start, 0);

    switch(kernel){
      case 1:
        inicializar_histograma<<<blocksPerGridM, threadsPerBlock>>>(h_device, M);
        if(cudaPeekAtLastError() != cudaSuccess) printf("inicializar_histograma<<<%lu, %lu>>> falla.\n", blocksPerGridM, threadsPerBlock);
        calcular_histograma <<<blocksPerGridN, threadsPerBlock>>>(v_device, h_device, N, M);
        if(cudaPeekAtLastError() != cudaSuccess) printf("calcular_histograma<<<%lu, %lu>>> falla.\n", blocksPerGridN, threadsPerBlock);
        break;
      default:
        printf("Cuidado! No se selecciona ningún kernel\n");
        break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if(i > 5) totalTime += (double)elapsedTime;
  }

  return totalTime / (R-6);
}
