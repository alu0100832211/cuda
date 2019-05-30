#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <cmath>


typedef unsigned int histogram_t;
typedef unsigned vector_t;

#define MIL 1000
#define MILLON MIL*MIL
#define N 80*MILLON
#define M 8                         //Tamaño histograma
#define TPB 1024                    //Threadsperblock
#define P (N+TPB-1)/TPB             //Nº sub-histogramas
#define R 106                       //Repeticiones para medir tiempo

#define SIZEH M*P*sizeof(histogram_t)
#define SIZEV N*sizeof(vector_t)

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

// Este kernel se llama con 1 histograma local por cada bloque, con N hilos.
__global__ void histograma_barrera(histogram_t * h, vector_t * v, 
    size_t m, size_t n){
  //Inicialización del histograma local
  size_t thread_id = threadIdx.x + blockIdx.x*blockDim.x;
  if(thread_id < n){ 
    __shared__ histogram_t smem [M];
    if(threadIdx.x < M) smem[threadIdx.x] = 0;
    __syncthreads();

    size_t h_pos = v[thread_id]%m;
    atomicAdd(&smem[h_pos], 1);
    __syncthreads();

    if(threadIdx.x < M)
      h[blockIdx.x*M + threadIdx.x] = smem[threadIdx.x];
  }
}

__global__ void reduccion_atomic(histogram_t * h, size_t m, size_t p){
  size_t thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(thread_id >= m*p || thread_id < m) return;
  atomicAdd(&h[thread_id%m], h[thread_id]);
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

void print_vector (histogram_t * h, size_t n);
void print_vector_device (histogram_t * h, size_t n);
int main (void){
  vector_t * v = (vector_t*)malloc(SIZEV);
  for(int i = 0; i < N; i++) v[i] = rand();
  vector_t * v_device;
  CUDA_CHECK_RETURN(cudaMalloc(&v_device, SIZEV));
  CUDA_CHECK_RETURN(cudaMemcpy(v_device, v, SIZEV, cudaMemcpyHostToDevice));

  histogram_t * h = (histogram_t*)malloc(SIZEH);
  histogram_t * h_device;
  CUDA_CHECK_RETURN(cudaMalloc(&h_device, SIZEH));

  printf("Llamando kernel con M = %i N = %i y P = %i\n", M, N, P);

  float elapsedTime = tiempo_kernel(h_device, v_device, 1);

  printf("Tiempo transcurrido: %f ms\n", elapsedTime);

  CUDA_CHECK_RETURN(cudaMemcpy(h, h_device, SIZEH, cudaMemcpyDeviceToHost));

  long long unsigned n_resultado = 0;
  for(int i = 0; i < M; i++){
    n_resultado+=h[i];
  }
  printf("Suma de elementos del histogr.: %llu\n", n_resultado);
  printf("Numero de elementos del vector: %i\n", N);

  free(v);
  cudaFree(v_device);
  free(h);
  cudaFree(h_device);
}

float tiempo_kernel(histogram_t * h, vector_t * v, unsigned kernel = 1){
  size_t threadsPerBlock = 1024;
  size_t blocksPerGridM = ((unsigned)M + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksPerGridN = ((unsigned)N + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksPerGridPM = ((unsigned)(P*M) + threadsPerBlock - 1) / threadsPerBlock;
  size_t blocksReduccion;
  
  cudaEvent_t start,stop, reduccion;
  cudaEventCreate(&start);
  cudaEventCreate(&reduccion);
  cudaEventCreate(&stop);
  float totalTime = 0, totalCalculo = 0, totalReduccion = 0;

  for(int i = 0; i < R; i++){
    size_t p = P;
    cudaEventRecord(start, 0);
    histograma_barrera <<<blocksPerGridN, threadsPerBlock>>>(h, v, M, N);
    cudaEventRecord(reduccion, 0);
    reduccion_atomic<<<blocksPerGridPM, threadsPerBlock>>>(h, M, p);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if(i > 5) totalTime += elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, reduccion);
    if(i > 5) totalCalculo += elapsedTime;
    cudaEventElapsedTime(&elapsedTime, reduccion, stop);
    if(i > 5) totalReduccion += elapsedTime;
  }
  printf("Tiempo de calculo %f ms ; tiempo reduccion %f ms \n ", totalCalculo/(R-6),totalReduccion/(R-6));

  return totalTime / (R-6);
}

void print_vector (histogram_t * h, size_t n){
  printf("n=%llu ", n);
  unsigned suma = 0;
  for(int i = 0; i < n; i++){
    suma += h[i];
    printf("%i ", h[i]);
  }
  printf("\nSuma: %u\n", suma);
}
void print_vector_device (histogram_t * h, size_t n){
  histogram_t * h_host = (histogram_t*)malloc(SIZEH);
  CUDA_CHECK_RETURN(cudaMemcpy(h_host, h, SIZEH, cudaMemcpyDeviceToHost)); 
  print_vector(h_host, n);
  free(h_host);
}
