
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 32
#define TILE_SIZE 16 
#define BATCH_SIZE 512
#define MAX_THREADS 1024
#define MAX_THREADS_PER_BLOC
#define MAX_THREADS_PER_BLOCK
namespace mxnet
{
namespace op
{
__constant__ float K_mask[8*8*8*16];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    extern __shared__ float shared_mem[];
    int X_tile_size = TILE_WIDTH + K - 1;
    
    //makeshift ceil function
    int W_grid = (int)((W_out-.5)/TILE_WIDTH+.5);
    
    //convenience variables
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    
    float * X_shared = (float*) &shared_mem[0];
//    float * K_shared = (float*) &shared_mem[X_tile_size*X_tile_size];
    
    int th = ty;
    int tw = tx;
    
    //determine tile start indices
    int w_base = (bz % W_grid) * TILE_WIDTH;
    int h_base = (bz / W_grid) * TILE_WIDTH;
    
    //assign each thread to an output element
    int w = w_base+tw;
    int h = h_base+th;
    
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) K_mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0;

    for(int c = 0; c < C; c++) {
        //load mask elements into shared memory
   //     if ((th < K) && (tw < K)) {
   //         K_shared[th*K+tw] = k4d(by, c, th, tw);
   //     }
        
   //     __syncthreads();

        //load input elements into shared memory
        for(int i = h; i < h_base + X_tile_size; i+= TILE_WIDTH){
            
            for(int j = w; j < w_base + X_tile_size; j += TILE_WIDTH){
                if(i < H && j < W){
                    X_shared[(i-h_base)*X_tile_size+(j-w_base)] = x4d(bx, c, i, j);
                } 
                else{
                    X_shared[(i-h_base)*X_tile_size+(j-w_base)] = 0.0f;
                }
            }
        }
        __syncthreads();
       
        
        //perform convolution
        for(int p = 0; p < K; p++){
            #pragma unroll 5
            for(int q = 0; q < K; q++){
                //acc += X_shared[(th+p)*X_tile_size+(tw+q)] * K_shared[p*K+q];
                acc += X_shared[(th+p)*X_tile_size+(tw+q)] * k4d(by, c, p, q);
            }
        }
        __syncthreads();
    }
    
     
    //only threads with indices in bounds contribute to final output
    if (h < H_out && w < W_out)
        y4d(bx, by, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
}

__global__ void tiledMatrixMultiply(float * A, float * B, float *C, int A_rows, int A_cols, int B_rows, int B_cols, int C_Rows, int C_Cols){
    __shared__ float subTileA[TILE_SIZE][TILE_SIZE];
    __shared__ float subTileB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by*TILE_SIZE+ty;
    int Col = bx*TILE_SIZE+tx;
    
    int sections = (int)(A_cols-0.5)/(TILE_SIZE+0.5);

    float res = 0.0f;
    for(int ph = 0; ph < sections; ++ph){
        if (Row < A_rows && (ph*TILE_SIZE+tx) < A_cols){
            subTileA[ty][tx] = A[Row*A_cols+ph*TILE_SIZE+tx];
        }
	else{
	    subTileA[ty][tx] = 0;
	}
        if(ph*TILE_SIZE+ty < B_rows && Col < B_cols){
            subTileB[ty][tx] = B[(ph*TILE_SIZE+ty)*B_cols+Col];
        }
	else{
	    subTileB[ty][tx] = 0;
	}

        __syncthreads();
        for(int k = 0; k < TILE_SIZE; ++k){
            res += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }

    if (Row < C_Rows && Col < C_Cols){
        C[Row*C_Cols+Col] = res;
    }
}

void matrixMult(int M, int C, int K, int H_out, int W_out, float *x, float *y, float *k){
    int A_rows = M;
    int A_columns = C * K * K;
    int B_columns = H_out * W_out;

    dim3 dimGrid(ceil(B_columns/(1.0*TILE_SIZE)), ceil(A_rows/(1.0*TILE_SIZE)), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    tiledMatrixMultiply<<<dimGrid, dimBlock>>>(x, y, k, A_rows, A_columns, A_columns, B_columns, A_rows, B_columns);
}

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a


__global__ void unroll_kernel(int C, int H, int W, int K, float * x, float * x_unroll){
    int c, s, h_out, w_out, h_unroll, w_base;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    int t = threadIdx.y + blockDim.y*blockIdx.y;
    int b = blockIdx.x;
    
    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll; 
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;

        #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                W_unroll = w_base + p*K + q;
                x_unroll[h_unroll*(H_out*W_out) + W_unroll] = x4d(b ,c, h_out+p, w_out+q);
            }
        }
    }
 
    #undef y4d
    #undef x4d
    #undef k4d
}
void unroll_gpu(int B, int C, int H, int W, int K, float * x, float * x_unroll){
    int H_out = H-K+1;
    int W_out = W-K+1;
   
    dim3 unrollGrid(B, ceil(C*W_out*H_out/(1.0*MAX_THREADS)),1) ;
    dim3 unrollBlock(1, MAX_THREADS, 1);

    unroll_kernel<<<unrollGrid, unrollBlock>>>(C, H, W, K, x, x_unroll);
}
/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; // batch index
    const int M = y.shape_[1]; // output number of channels
    const int C = x.shape_[1]; // input number of channels 
    const int H = x.shape_[2]; // input rows
    const int W = x.shape_[3]; // input columns
    const int K = w.shape_[3]; // kernel width (assuming square kernel)
    
    const int W_out = W-K+1;  // output columns
    const int H_out = H-K+1;  // output rows
    
    const int W_grid = ceil((W_out)/TILE_WIDTH)+1;
    const int H_grid = ceil((H_out)/TILE_WIDTH)+1;
    
    const int Z = W_grid*H_grid;
    //printf("Initializing convolution with params: \nB\t: %d\nM\t: %d\nC\t: %d\nH\t: %d\nW\t: %d\nK\t: %d\n", B, M, C, H, W, K);
    //printf("X_tile_width: %d\n", TILE_WIDTH+K-1);
    
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //printf("Launching Thread Grid with: \ngridDim: x\t: %d\t y\t: %d\tz\t: %d\n", gridDim.x, gridDim.y, gridDim.z); 
    //printf("blockDim: x\t: %d\ty\t: %d\tz\t: %d\n", blockDim.x, blockDim.y, blockDim.z);
    
    float * x_unrolled;

    cudaMalloc((void**)&x_unrolled, sizeof(float)*C*K*K*W_out*H_out);

    size_t shmem_size = sizeof(float)*(TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+(K*K)*sizeof(float);

    unroll_gpu(B, C, H, W, K, x.dptr_, x_unrolled);

    matrixMult(M, C, K, H_out, W_out, x_unrolled, y.dptr_, w.dptr_);
    //int mask_size = w.
    //cudaMemcpyToSymbol(K_mask, w.dptr_, mask_size*sizeof(float);

    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
//    MSHADOW_CUDA_CALL(cudaDeviceReset());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    //CHECK_EQ(0,1) << "Remove th:is line and replace it with your implementation.";
}
}
}

#endif
