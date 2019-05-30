#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <cmath>


typedef unsigned int histogram_t;
typedef unsigned vector_t;

#define MIL 1000
#define MILLON MIL*MIL
#define N 20*MILLON
#define M 8                         //Tamaño histograma
#define P 10                        //Nº sub-histogramas
#define Q (int)ceil((float)N/(float)P)   //Elementos por histograma
#define R 101                       //Repeticiones para medir tiempo

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
  if(thread_id < n) h[thread_id] = 0;
}
__global__ void p_histogramas(histogram_t * h, vector_t * v, 
    size_t m, size_t n, size_t q){
    size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_id < n){
      size_t histogram_id = thread_id/q;
      size_t histogram_pos = v[thread_id] % m;
      histogram_t * addr = &h[histogram_id*m+histogram_pos];
      atomicAdd(addr, 1);
    }
}

__global__ void reduccion_atomic(histogram_t * h, size_t m, size_t p){
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(thread_id >= m && thread_id < m*p){
    size_t pos = thread_id%m;
    atomicAdd(&h[pos], h[thread_id]);
  }
}

//Hay un hilo por cada dos posiciones del histograma m*(p/2)
__global__ void reduccion_paralela(histogram_t * h, size_t m, size_t p){
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(thread_id >= m*(p/2)) return;
  h[thread_id] += h[thread_id + m*(p/2)];
  if((p%2) && thread_id < m)
    h[thread_id] += h[thread_id + 2*m*(p/2)];
}

float tiempo_kernel(histogram_t * h, vector_t * v, unsigned kernel);

int main (void){
  int sizeh = P*M*sizeof(histogram_t);
  histogram_t * h = (histogram_t*)malloc(sizeh);
  int sizev = N*sizeof(vector_t);
  vector_t * v = (vector_t*)malloc(sizev);
  for(int i = 0; i < N; i++) v[i] = rand();
  histogram_t * h_device;
  CUDA_CHECK_RETURN(cudaMalloc(&h_device, sizeh));
  vector_t * v_device;
  CUDA_CHECK_RETURN(cudaMalloc(&v_device, sizev));
  CUDA_CHECK_RETURN(cudaMemcpy(v_device, v, sizev, cudaMemcpyHostToDevice));

  printf("Llamando kernel con M %i N %i Q %i\n", M, N, Q);

  float elapsedTime = tiempo_kernel(h_device, v_device, 1);

  printf("Tiempo transcurrido: %f ms\n", elapsedTime);

  CUDA_CHECK_RETURN(cudaMemcpy(h, h_device, sizeh, cudaMemcpyDeviceToHost));


  printf("Resultado: ");
  long long unsigned n_resultado = 0;
  for(int i = 0; i < M; i++){
    n_resultado+=h[i];
    printf("%llu + ", h[i]);
  }
  printf("= %llu\n", n_resultado);
  printf("Nº de elementos del vector: %i\n", N);

  free(h);
  free(v);
  
  cudaFree(v_device);
  cudaFree(h_device);
}

float tiempo_kernel(histogram_t * h, vector_t * v, unsigned kernel){
  size_t threadsPerBlock = 1024;
  size_t blocksPerGridM = ((unsigned)M + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksPerGridN = ((unsigned)N + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksPerGridPM = ((unsigned)(P*M) + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksReduccion;
  
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float totalTime = 0;

  for(int i = 0; i < R; i++){
    size_t p = P;
    cudaEventRecord(start, 0);

    switch(kernel){
      case 1:
        inicializar_histograma <<<blocksPerGridPM, threadsPerBlock>>>(h, M*P);
        p_histogramas <<<blocksPerGridN, threadsPerBlock>>>(h, v, M, N, Q);
        //reduccion_atomic <<<blocksPerGridPM, threadsPerBlock>>>(h, M, P);
        while(p > 1){
          blocksReduccion = M*(p/2);
          reduccion_paralela<<<blocksReduccion, threadsPerBlock>>>(h, M, p);
          p/=2;
        }
        if(cudaPeekAtLastError() != cudaSuccess) printf("Falla para %lu hilos/bloque y %lu bloques\n", threadsPerBlock, blocksPerGridPM);
        break;
      default:
        printf("Cuidado! No se selecciona ningún kernel\n");
        break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    CUDA_CHECK_RETURN(cudaPeekAtLastError());
    CUDA_CHECK_RETURN(cudaGetLastError());

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if(i != 0) totalTime += elapsedTime;
  }
  return totalTime / (R-1);
}
