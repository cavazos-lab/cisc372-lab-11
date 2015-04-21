#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

const int N=1024;  /* multiple blocks used, change back from 512 to 1024 */

double rtclock()
{
   struct timezone Tzp;
   struct timeval Tp;

   int stat;
   stat = gettimeofday (&Tp, &Tzp);

   if (stat != 0) printf("Error return from gettimeofday: %d",stat);

   return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__ void matrixAdd(double *A, double *B, double *C, int N)
{
    int i, j;
    i = 0;
    j = blockIdx.x*blockDim.x + threadIdx.x; 

    if (j < N )
    {
      // The following statement is essentially: C[i][j] = A[i][j] + B[i][j]
      C[i*N+j] = A[i*N+j] + B[i*N+j];

      // Can also use the following
      //C[j*N+i] = A[j*N+i] + B[j*N+i];
    }
}

int main(int argc, char*argv[])
{
    double A[1][N];
    double *B;      /* Program Stack Size is limited, so only Array A could be statically allocated (on the stack) */
    double *C;      /* B and C arrays are to be dynamically allocated on the heap, which has much larger space */
	double *d_A, *d_B, *d_C;
    double * gpu_C;          /* stores the copy of d_C because CPU cannot access d_C directly
                                copy via cudaMemcpy Device (i.e. d_C) -> Host (i.e. gpu_C) */

    int size = 1 * N * sizeof (double);
    B = (double *) malloc (size);

    C = (double *) malloc (size);

    int THREAD_DIMX,THREAD_DIMY,BLOCK_DIMX,BLOCK_DIMY;
    
    
    gpu_C= (double*) malloc (size);

	/* allocate space for device copies */
	cudaMalloc( (void **) &d_A, size );
	cudaMalloc( (void **) &d_B, size );
    cudaMalloc( (void **) &d_C, size);

	for( int i = 0; i < 1; i++ )
        for( int j = 0; j < N; j++ )
	{
        A[i][j] = 1.0;
        B[i*N+j] = 2.0;
	}

	/* copy inputs to device */
	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
    THREAD_DIMX = 32; 
    THREAD_DIMY = 1;

    BLOCK_DIMX = N/32;
    BLOCK_DIMY = 1; 

    dim3 dimGrid(BLOCK_DIMX,BLOCK_DIMY,1);
    dim3 dimBlock(THREAD_DIMX,THREAD_DIMY,1);
    
    double start_cpu = rtclock();

	matrixAdd<<< dimGrid, dimBlock>>>( d_A, d_B, d_C, N);

    cudaThreadSynchronize();


    double end_cpu = rtclock();
    printf("total time is %lf\n",(double)(end_cpu-start_cpu));  

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy( gpu_C, d_C, size, cudaMemcpyDeviceToHost );

    for (int i=0; i<1; i++)
        for (int j=0; j<N; j++)
        {
           C[i*N+j] = A[i][j] + B[i*N+j];
        if ( abs(C[i*N+j] - gpu_C[i*N+j]) > 1e-5 )
        {
            printf("CPU %f and GPU %f results do not match!\n", C[i*N+j], gpu_C[i*N+j]);
            exit(-1);
        } 
        }


	/* clean up */

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );
    free(gpu_C);
	
	return 0;
} /* end main */
