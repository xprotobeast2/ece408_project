#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <mxnet/base.h>
#define TILE_WIDTH 32
#define MULT_TILE_WIDTH 32
#define FUSION_TILE 16
#define BATCH_SIZE 512
#define MAX_THREADS 1024
#define BUFFER 8*8*12
#define MAX_THREADS 1024
namespace mxnet
{
    namespace op
    {
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
            float * K_shared = (float*) &shared_mem[X_tile_size*X_tile_size];

            int th = ty;
            int tw = tx;

            //determine tile start indices
            int w_base = (bz % W_grid) * TILE_WIDTH;
            int h_base = (bz / W_grid) * TILE_WIDTH;

            //assign each thread to an output element
            //int w = w_base+tw;
            //int h = h_base+th;
            int w = w_base + tw;
            int h = h_base + th;
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            float acc = 0;
            for(int c = 0; c < C; c++) {
                //load mask elements into shared memory
                if ((th < K) && (tw < K)) {
                    K_shared[th*K+tw] = k4d(by, c, th, tw);
                }

                __syncthreads();
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
                    for(int q = 0; q < K; q++){
                        acc += X_shared[(th+p) * X_tile_size + (tw+q)] * K_shared[p * K + q];
                    }
                }
            }
        }

        __global__ void forward2_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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
            float * K_shared = (float*) &shared_mem[X_tile_size*X_tile_size];

            int th = ty;
            int tw = tx;

            //determine tile start indices
            int w_base = (bz % W_grid) * TILE_WIDTH;
            int h_base = (bz / W_grid) * TILE_WIDTH;

            //assign each thread to an output element
            //int w = w_base+tw;
            //int h = h_base+th;
            int w = w_base + tw;
            int h = h_base + th;
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            float acc = 0;
            for(int c = 0; c < C; c++) {
                //load mask elements into shared memory
                if ((th < K) && (tw < K)) {
                    K_shared[th*K+tw] = k4d(by, c, th, tw);
                }

                __syncthreads();
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
                    for(int q = 0; q < K; q++){
                        acc += X_shared[(th+p) * X_tile_size + (tw+q)] * K_shared[p * K + q];
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


__global__ void unroll_input(float *y, const float *x, float *X_unroll, const int B, const int M, const int C, const int H, const int W, const int K)
{
    
    int c, s, h_out, w_out, h_unroll, w_base, p, q;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;

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
                int w_unroll = w_base + p*K + q;
                 X_unroll[h_unroll*(H_out*W_out) + w_unroll] = x4d(b ,c, h_out+p, w_out+q);
            }
        }
    }
 
    #undef y4d
    #undef x4d
    #undef k4d
}




// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
        //__constant__ float w_mask[12 * 12 * 12 * 12];
        __global__ void forward3_kernel(const int C, const int M, const int H, const int W, const int K, const float * x, const float * k, float * y){
            __shared__ float A[FUSION_TILE][FUSION_TILE];
            __shared__ float B[FUSION_TILE][FUSION_TILE];
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0 + blockIdx.z * (C * H * W)]
            #define k2d(i1, i0) k[(i1) * (C  * K * K) + (i0)]
            #define y2d(i1, i0) y[(i1) * (H_out * W_out) + (i0) + blockIdx.z * (M * H_out * W_out)]

            int row = blockIdx.y * FUSION_TILE + threadIdx.y;
            int col = blockIdx.x * FUSION_TILE + threadIdx.x;
            float sum = 0.0f;
            
            for(int tile_idx = 0; tile_idx < ceil( (C * K * K) / (1.0 * FUSION_TILE)); tile_idx++) {
                if ((row < M) && (tile_idx * FUSION_TILE + threadIdx.x) < (C * K * K)){
                    A[threadIdx.y][threadIdx.x] = k2d(row, tile_idx * FUSION_TILE + threadIdx.x);
                }
                else {
                    A[threadIdx.y][threadIdx.x] = 0.0f;
                }

                int section_idx = threadIdx.y + tile_idx * FUSION_TILE;
                if (col < H_out * W_out && section_idx < C * K * K) {
                    B[threadIdx.y][threadIdx.x] = x3d((section_idx)/(K*K), col/W_out + (section_idx/K) % K, col % W_out + section_idx % K);
                }
                else {
                    B[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();
                for(int z = 0; z < FUSION_TILE; z++){
                    sum += A[threadIdx.y][z] * B[z][threadIdx.x];
                }
                __syncthreads();
            }
            
            if (row < M && col < H_out * W_out){
                y2d(row, col) = sum;
            }
 
            #undef x3d
            #undef k2d
            #undef y2d
        }

         #define FUSION_TILE_X 16
         #define FUSION_TILE_Y 8
         #define RATIO_TILE (FUSION_TILE_X/FUSION_TILE_Y)
        /* 
         __global__ void fusion2_kernel(const int C, const int M, const int H, const int W, const int K, const float * x, const float * k, float * y){
            __shared__ float B[FUSION_TILE_X][FUSION_TILE_X];
            //float out[FUSION_TILE_X] = {};
            int H_out, W_out, sec, s;
            H_out = H - K + 1;
            W_out = W - K + 1;
            #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0 + blockIdx.z * (C * H * W)]
            #define k2d(i1, i0) k[(i1) * (C  * K * K) + (i0)]
            #define y2d(i1, i0) y[(i1) * (H_out * W_out) + (i0) + blockIdx.z * (M * H_out * W_out)]
            const unsigned row_base = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned col_base = blockIdx.x * blockDim.x + threadIdx.x;
            float reg_w;
            float sum = 0.0f;
            for (unsigned tile_idx = 0; tile_idx < ceil ( (C * K * K) / (1.0 * FUSION_TILE_X)); tile_idx++) {
                int offset = tile_idx * FUSION_TILE_X;
                int section_idx = tile_idx * FUSION_TILE_Y + threadIdx.y;
                if (section_idx+offset < C * K * K && col_base < H_out * W_out) {
                        B[section_idx][threadIdx.x] = x3d((section_idx+offset)/(K * K), (col_base/W_out) + (section_idx+offset)%(K * K)/K, (col_base % W_out) + ((section_idx+offset)%(K*K))%K);
                    } else {
                        B[section_idx][threadIdx.x] = 0.0f;
 }
                
                

                if (row_base < M && offset + threadIdx.x < C * K * K) {
                    reg_w = k2d(row_base, offset + threadIdx.x);
                } else {
                    reg_w = 0.0f;
                }
                __syncthreads();
                for (s = 0; s < FUSION_TILE_X; s++) {
                         sum += reg_w * B[s][threadIdx.x];
                    }
                }
            


                if (row_base < M && col_base < H_out * W_out)
                    y2d(row_base, col_base) = sum;
                

#undef x3d
#undef k2d
#undef y2d
         }*/

        __global__ void unroll_inputs(const int C, const int H, const int W, const int K, const int B, float * x, float * x_unrolled) {
            int c, s, h_out, w_out, h_unroll, w_base;
            int cur_row, cur_col;
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            int W_unroll = H_out * W_out;

            int t = threadIdx.x + blockDim.x * blockIdx.x;
            if (t < C * K * K * H_out * W_out) {
                c = t / W_unroll;
                s = t % W_unroll; 
                h_out = s / W_out;
                w_out = s % W_out;
                h_unroll = h_out * W_out + w_out;
                w_base = c * K * K;
                cur_col = c % K;
                cur_row = (c/K) % K;
                int to_out = (c/K) / K;
                #define x3d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0 + B * C * H * W]
                #define x1d(i0) x_unrolled[i0]
                x1d(t) = x3d(to_out, h_out + cur_row, w_out + cur_col);
            }
                __syncthreads();
                #undef x3d
                #undef x1d
        }

        __global__ void unroll_weights_kernel(const int M, const int C, const int K, const float * w, float * w_unrolled) {

        }


        __global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
            __shared__ float M[MULT_TILE_WIDTH][MULT_TILE_WIDTH];
            __shared__ float N[MULT_TILE_WIDTH][MULT_TILE_WIDTH];
            
            // Need to linearize the block matrix
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // Row and Col of C's element that is being worked on
            int Row = by*MULT_TILE_WIDTH + ty;
            int Col = bx*MULT_TILE_WIDTH + tx;

            float sum = 0.0;
            // The loop will be over a linearized tile index
            for (int tile_x = 0; tile_x < ceil(numAColumns/(float)MULT_TILE_WIDTH); tile_x++) {

                // First load the values of M and N that this thread is reponsible for
                if ((Row < numARows) && (tile_x*MULT_TILE_WIDTH + tx) < numAColumns) {
                    M[ty][tx] = A[Row*numAColumns + tile_x*MULT_TILE_WIDTH + tx];      
                } else {
                    M[ty][tx] = 0.0;
                }
                //__syncthreads();

                if((Col < numBColumns) && (tile_x*MULT_TILE_WIDTH + ty) < numBRows) {
                    N[ty][tx] = B[(tile_x*MULT_TILE_WIDTH + ty)*numBColumns + Col];
                } else {
                    N[ty][tx] = 0.0;
                }
                // Make sure all threads in block have loaded their values 
                __syncthreads();
                for (int k = 0; k < MULT_TILE_WIDTH; k++) {
                    sum += M[ty][k]*N[k][tx];
                }
                __syncthreads();
            }
            if (Row < numCRows && Col < numCColumns) {
                C[Row*numCColumns + Col] = sum;
            }
        }

        __global__ void reroll_output_kernel(int M, int H_out, int W_out, float * y_unrolled, float * y) {

        }

        /*
           Host launchers

           - unroll input
           - unroll weights
           - multiply input x weights = output
           - reroll output
         */

        void fusion_launcher(int C, int M, int H, int W, int K, int B, float * x, float *w, float *y) {
            int A_rows = M;
            int B_cols = (H-K+1)*(W-K+1);
            dim3 dimGrid(ceil(B_cols/(1.0*FUSION_TILE)), ceil(A_rows/(1.0*FUSION_TILE)), B);
            dim3 dimBlock(FUSION_TILE, FUSION_TILE, 1);
            forward3_kernel<<<dimGrid, dimBlock>>>(C, M, H, W, K, x, w, y);
        }
        
        void fusion2_launcher(int C, int M, int H, int W, int K, int B, float *x , float *w, float *y) {
            dim3 dimGrid(ceil(((H - K + 1) * (W - K + 1))/(1.0 * FUSION_TILE)), ceil(M/(1.0 * FUSION_TILE)), B);
            dim3 dimBlock(FUSION_TILE, FUSION_TILE, 1);
            forward3_kernel<<<dimGrid, dimBlock>>>(C, M, H, W, K, x, w, y);
        }

        void unroll_input(int C, int H, int W, int K, int B, float * x, float * x_unrolled) {
            int H_out = H-K+1;
            int W_out = W-K+1;
            int mask_unroll_size = C * K * K;
            int W_unroll = W_out * H_out;
            int num_threads = mask_unroll_size * W_unroll;

            dim3 unrollGrid(ceil(num_threads/(1.0*(MAX_THREADS))), 1, 1);
            dim3 unrollBlock(ceil(MAX_THREADS/(1.0 )), 1, 1);

            unroll_inputs<<<unrollGrid, unrollBlock>>>(C, H, W, K, B, x, x_unrolled);
        }

        void unroll_weights(int M, int C, int K, float * w, float * w_unrolled) {

            dim3 unrollGrid(ceil((C*K*K*M)/(1.0*MAX_THREADS)),1,1);
            dim3 unrollBlock(MAX_THREADS,1,1);

            unroll_weights_kernel<<<unrollGrid, unrollBlock>>>(M, C, K, w, w_unrolled);

        }

        void matrixMult(int M, int C, int K, int b, int H_out, int W_out, float *w, float *x, float *y) { 
            int A_rows = M;
            int A_columns = C * K * K;
            int B_columns = H_out * W_out;

            dim3 dimGrid(ceil(B_columns/(1.0*MULT_TILE_WIDTH)), ceil(A_rows/(1.0*MULT_TILE_WIDTH)), b);
            dim3 dimBlock(MULT_TILE_WIDTH, MULT_TILE_WIDTH, 1);
            matrixMultiplyShared<<<dimGrid, dimBlock>>>(w, x, y, A_rows, A_columns, A_columns, B_columns, A_rows, B_columns);
        }

        void reroll_output(int M, int H_out, int W_out, int B, float * y_unrolled, float * y) {
            dim3 rerollGrid(ceil((M*H_out*W_out)/(1.0*MAX_THREADS)), 1, B);
            dim3 rerollBlock(MAX_THREADS, 1,1);

            reroll_output_kernel<<<rerollGrid, rerollBlock>>>(M, H_out, W_out, y_unrolled, y);
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

                dim3 gridDim(B, M, Z);
                dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);


                // device memory initializations
                //float * x_unrolled;
                //float * w_unrolled;
                //float * y_unrolled;

                // Allocate device memory
                //cudaMalloc((void**)&x_unrolled, sizeof(float)*C*K*K*W_out*H_out);
                //cudaMalloc((void**)&w_unrolled, sizeof(float)*M*C*K*K);
                //cudaMalloc((void**)&y_unrolled, sizeof(float)*M*W_out*H_out);

                //size_t shmem_size = sizeof(float)*(TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+(K*K)*sizeof(float);
                //forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

                // Loop over all elements in the batch and call unroll functions

                /*  for (int b = 0; b <= B; b++) {
                    unroll_input(C, H, W, K, b, x.dptr_, x_unrolled);
                //unroll_weights(M,C, K, w.dptr_, w_unrolled);
                //matrixMult(M, C, K, H_out, W_out, w_unrolled, x_unrolled, y.dptr_+b*M*H_out*W_out);
                matrixMult(M, C, K, b, H_out, W_out, w.dptr_, x_unrolled, y.dptr_);
                //reroll_output(M, H_out, W_out, b, y_unrolled, y.dptr_);
                fusion_launcher(C, M, H, W, K, B, x.dptr_, x_unrolled, w.dptr_, y.dptr_);
                }
                 */
                
                /* I wanted to store mask values in constant memory. 
                //cudaMemcpyToSymbol(w_mask, w.dptr_, C * M * K * K * sizeof(float), 0, cudaMemcpyHostToDevice); */
                fusion_launcher(C, M, H, W, K, B, x.dptr_, w.dptr_, y.dptr_);
//                unroll_input(C, H, W, K, B, x.dptr_, x_unrolled);
//                matrixMult(M, C, K, B, H_out, W_out, w.dptr_, x_unrolled, y.dptr_);
                //forward2_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
                //cudaMemcpyToSymbol(k_mask, w.dptr_, C*K*K*M*sizeof(float),0, cudaMemcpyHostToDevice);

                //forward_kernel_const<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
                //cudaFree(x_unrolled);
                //cudaFree(w_unrolled);
                //cudaFree(y_unrolled);
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
