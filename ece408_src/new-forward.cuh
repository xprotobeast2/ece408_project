#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <mxnet/base.h>

// shared Mem convolution
#define TILE_WIDTH 32

// For register tiling 
#define TILE_WIDTH_W 64
#define TILE_WIDTH_X 16
#define TILE_WIDTH_RATIO 4

// For matrix multplication and fusion kernel

#define MULT_TILE_WIDTH 32
#define FUSION_TILE 16

//For unrolling and kernel const mem optimizations
#define MAX_THREADS 1024
#define BUFFER 8*8*12



namespace mxnet
{
    namespace op
    {

        /*
            void shared_conv_kernel()

                Implements forward convolution using tiling and shared memory

        */

        __global__ void shared_conv_kernel(float *y, const float *x, const float *k, 
                                            int B, int C, int M, int H, int W, int K)
        {
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

            //only threads with indices in bounds contribute to final output
            if (h < H_out && w < W_out) {
                y4d(bx, by, h, w) = acc; 
            }

            #undef x4d
            #undef k4d
            #undef y4d
        }


        /*
            void fusion_kernel     
            
                Implements forward convolution as a general matrix multiplication
                Matrix multiplication uses tiling and shared memory
        */

        __global__ void fusion_kernel(float * y, const float * x, const float * k, 
                                                int C, int M, int H, int W, int K){

            __shared__ float W_tile[FUSION_TILE][FUSION_TILE];
            __shared__ float X_tile[FUSION_TILE][FUSION_TILE];
            
            // Define output dims
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            
            // Macros for tensor access
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            
            // This is the row and column in the output that this thread is calculating
            int row = blockIdx.y * FUSION_TILE + threadIdx.y;
            int col = blockIdx.x * FUSION_TILE + threadIdx.x;
            
            // Accumulator
            float sum = 0.0f;
            
    
            for(int tile_idx = 0; tile_idx < ceil( (C * K * K) / (1.0 * FUSION_TILE)); tile_idx++) {
                
                int section_row = tile_idx * FUSION_TILE + threadIdx.y;
                int section_col = tile_idx * FUSION_TILE + threadIdx.x;

                if ((row < M) && (section_col < (C * K * K))){
                    W_tile[threadIdx.y][threadIdx.x] = k4d(row, (section_col/(K*K)), ((section_col%(K*K))/K), ((section_col%(K*K))%K));
                }
                else {
                    W_tile[threadIdx.y][threadIdx.x] = 0.0f;
                }

                if ((col < (H_out * W_out)) && (section_row < (C * K * K))) {
                    X_tile[threadIdx.y][threadIdx.x] = x4d(blockIdx.z, (section_row/(K*K)),((col/W_out)+(section_row%(K*K))/K), ((col%W_out) + (section_row%(K*K))%K));
                }
                else {
                    X_tile[threadIdx.y][threadIdx.x] = 0.0f;
                }
                __syncthreads();
                for(int z = 0; z < FUSION_TILE; z++){
                    sum += W_tile[threadIdx.y][z] * X_tile[z][threadIdx.x];
                }
                __syncthreads();
            }
            
            if ((row < M) && (col < H_out * W_out)){
                y4d(blockIdx.z, row, (col/W_out), (col%W_out)) = sum;
            }
 
            #undef x4d
            #undef k4d
            #undef y4d
        }

        /*
            void register_tiled_fusion_kernel     
            
                Implements forward convolution as a general matrix multiplication
                Using joint register and shared memory tiling
        */

        __global__ void register_tiled_fusion_kernel(float * y, const float * x, const float * k, 
                                                int C, int M, int H, int W, int K){

            __shared__ float X_tile[TILE_WIDTH_RATIO][TILE_WIDTH_X];
            
            // Define output dims
            int H_out = H - K + 1;
            int W_out = W - K + 1;
            
            // Macros for tensor access
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
            

            // row and col indices for tiles that remain constant throughout thread work
            int w_tile_row = blockIdx.y * TILE_WIDTH_W + threadIdx.x;
            int x_tile_col_base = blockIdx.x * TILE_WIDTH_X;
            int x_tile_col = x_tile_col_base + (threadIdx.x%TILE_WIDTH_X);

            
            // TILE_WIDTH_X registers for output, and TILE_WIDTH_RATIO registers for kernel elements

            float p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0, p5 = 0.0, p6 = 0.0, p7 = 0.0;
            float p8 = 0.0, p9 = 0.0, p10 = 0.0, p11 = 0.0, p12 = 0.0, p13 = 0.0, p14 = 0.0, p15 = 0.0;
            float w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;

            // Macros for loop unrolling

            for(int tile_idx = 0; tile_idx < ceil( (C * K * K) / (1.0 * TILE_WIDTH_RATIO)); tile_idx++) {
                
                // Compute the base of the corners of the iteration dependent tiles
                int x_tile_row = tile_idx * TILE_WIDTH_RATIO + (threadIdx.x/TILE_WIDTH_X);
                int w_tile_col_base = tile_idx * TILE_WIDTH_RATIO;
                
                // Collaboratively load TILE_WIDTH_M into X_tile
                if ((x_tile_col < (H_out * W_out)) && (x_tile_row < (C * K * K))) {
                    X_tile[threadIdx.x/TILE_WIDTH_X][threadIdx.x%TILE_WIDTH_X] = x4d(blockIdx.z, (x_tile_row/(K*K)),((x_tile_col/W_out)+(x_tile_row%(K*K))/K), ((x_tile_col%W_out) + (x_tile_row%(K*K))%K));
                } 
                else {
                    X_tile[threadIdx.x/TILE_WIDTH_X][threadIdx.x%TILE_WIDTH_X] = 0.0f;
                }
                
                // Load TILE_WIDTH_RATIO elements from W into registers 
                if (w_tile_row < M) {

                    if (w_tile_col_base < (C * K * K)) {
                        w0 = k4d(w_tile_row, (w_tile_col_base/(K*K)), (w_tile_col_base%(K*K))/K, (w_tile_col_base%(K*K))%K);
                    }
                    else {
                        w0 = 0.0f;
                    }
                    if (w_tile_col_base + 1 < (C * K * K)) {
                        w1 = k4d(w_tile_row, ((w_tile_col_base + 1)/(K*K)), ((w_tile_col_base + 1)%(K*K))/K, ((w_tile_col_base + 1)%(K*K))%K);
                    }
                    else {
                        w1 = 0.0f;
                    }
                    if (w_tile_col_base + 2 < (C * K * K)) {
                        w2 = k4d(w_tile_row, ((w_tile_col_base + 2)/(K*K)), ((w_tile_col_base + 2)%(K*K))/K, ((w_tile_col_base + 2)%(K*K))%K);
                    }
                    else {
                        w2 = 0.0f;
                    }
                    if (w_tile_col_base + 3 < (C * K * K)) {
                        w3 = k4d(w_tile_row, ((w_tile_col_base + 3)/(K*K)), ((w_tile_col_base + 3)%(K*K))/K, ((w_tile_col_base + 3)%(K*K))%K);
                    }
                    else {
                        w3 = 0.0f;
                    }
                }
                
                __syncthreads();

                p0 += w0 * X_tile[0][0];
                p1 += w0 * X_tile[0][1];
                p2 += w0 * X_tile[0][2];
                p3 += w0 * X_tile[0][3];
                p4 += w0 * X_tile[0][4];
                p5 += w0 * X_tile[0][5];
                p6 += w0 * X_tile[0][6];
                p7 += w0 * X_tile[0][7];
                p8 += w0 * X_tile[0][8];
                p9 += w0 * X_tile[0][9];
                p10 += w0 * X_tile[0][10];
                p11 += w0 * X_tile[0][11];
                p12 += w0 * X_tile[0][12];
                p13 += w0 * X_tile[0][13];
                p14 += w0 * X_tile[0][14];
                p15 += w0 * X_tile[0][15];

                p0 += w1 * X_tile[1][0];
                p1 += w1 * X_tile[1][1];
                p2 += w1 * X_tile[1][2];
                p3 += w1 * X_tile[1][3];
                p4 += w1 * X_tile[1][4];
                p5 += w1 * X_tile[1][5];
                p6 += w1 * X_tile[1][6];
                p7 += w1 * X_tile[1][7];
                p8 += w1 * X_tile[1][8];
                p9 += w1 * X_tile[1][9];
                p10 += w1 * X_tile[1][10];
                p11 += w1 * X_tile[1][11];
                p12 += w1 * X_tile[1][12];
                p13 += w1 * X_tile[1][13];
                p14 += w1 * X_tile[1][14];
                p15 += w1 * X_tile[1][15];

                p0 += w2 * X_tile[2][0];
                p1 += w2 * X_tile[2][1];
                p2 += w2 * X_tile[2][2];
                p3 += w2 * X_tile[2][3];
                p4 += w2 * X_tile[2][4];
                p5 += w2 * X_tile[2][5];
                p6 += w2 * X_tile[2][6];
                p7 += w2 * X_tile[2][7];
                p8 += w2 * X_tile[2][8];
                p9 += w2 * X_tile[2][9];
                p10 += w2 * X_tile[2][10];
                p11 += w2 * X_tile[2][11];
                p12 += w2 * X_tile[2][12];
                p13 += w2 * X_tile[2][13];
                p14 += w2 * X_tile[2][14];
                p15 += w2 * X_tile[2][15];

                p0 += w3 * X_tile[3][0];
                p1 += w3 * X_tile[3][1];
                p2 += w3 * X_tile[3][2];
                p3 += w3 * X_tile[3][3];
                p4 += w3 * X_tile[3][4];
                p5 += w3 * X_tile[3][5];
                p6 += w3 * X_tile[3][6];
                p7 += w3 * X_tile[3][7];
                p8 += w3 * X_tile[3][8];
                p9 += w3 * X_tile[3][9];
                p10 += w3 * X_tile[3][10];
                p11 += w3 * X_tile[3][11];
                p12 += w3 * X_tile[3][12];
                p13 += w3 * X_tile[3][13];
                p14 += w3 * X_tile[3][14];
                p15 += w3 * X_tile[3][15];

                __syncthreads();
            }
            
            if (w_tile_row < M) {

                if(x_tile_col_base < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, (x_tile_col_base/W_out), (x_tile_col_base%W_out)) = p0;
                }
                if(x_tile_col_base + 1 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 1)/W_out), ((x_tile_col_base + 1)%W_out)) = p1;
                }
                if(x_tile_col_base + 2 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 2)/W_out), ((x_tile_col_base + 2)%W_out)) = p2;
                }
                if(x_tile_col_base + 3 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 3)/W_out), ((x_tile_col_base + 3)%W_out)) = p3;
                }
                if(x_tile_col_base + 4 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 4)/W_out), ((x_tile_col_base + 4)%W_out)) = p4;
                }
                if(x_tile_col_base + 5 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 5)/W_out), ((x_tile_col_base + 5)%W_out)) = p5;
                }
                if(x_tile_col_base + 6 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 6)/W_out), ((x_tile_col_base + 6)%W_out)) = p6;
                }
                if(x_tile_col_base + 7 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 7)/W_out), ((x_tile_col_base + 7)%W_out)) = p7;
                }
                if(x_tile_col_base + 8 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 8)/W_out), ((x_tile_col_base + 8)%W_out)) = p8;
                }
                if(x_tile_col_base + 9 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 9)/W_out), ((x_tile_col_base + 9)%W_out)) = p9;
                }
                if(x_tile_col_base + 10 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 10)/W_out), ((x_tile_col_base + 10)%W_out)) = p10;
                }
                if(x_tile_col_base + 11 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 11)/W_out), ((x_tile_col_base + 11)%W_out)) = p11;
                }
                if(x_tile_col_base + 12 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 12)/W_out), ((x_tile_col_base + 12)%W_out)) = p12;
                }
                if(x_tile_col_base + 13 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 13)/W_out), ((x_tile_col_base + 13)%W_out)) = p13;
                }
                if(x_tile_col_base + 14 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 14)/W_out), ((x_tile_col_base + 14)%W_out)) = p14;
                }
                if(x_tile_col_base + 15 < (H_out * W_out)) {
                    y4d(blockIdx.z, w_tile_row, ((x_tile_col_base + 15)/W_out), ((x_tile_col_base + 15)%W_out)) = p15;
                }
            }
 
            #undef x4d
            #undef k4d
            #undef y4d
        }


        /*
            void unroll_input

                This kernel takes the input B x C x H x W input and converts it into a 2D (C*K*K) x ((H - K + 1)*(W - K +1)) matrix

                in the fusion kernel, there is no input memory coalescing
        
        */

        __global__ void unroll_kernel(const float *x, float *x_unrolled, int batchIdx, int C, int H, int W, int K)
        {
            
            int c, s, h_out, w_out, h_unroll, w_base;

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int W_unroll = H_out * W_out;

            int t = threadIdx.y + blockDim.y*blockIdx.y;

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
                         x_unrolled[h_unroll*(H_out*W_out) + w_unroll] = x4d(batchIdx ,c, h_out+p, w_out+q);
                    }
                }
            }
         
            #undef x4d
        }

        __global__ void matmult_kernel(const float *w, float *x_unrolled, float *y, int numWRows, int numWCols, int numXRows, int numXCols) {
            
            __shared__ float M[MULT_TILE_WIDTH][MULT_TILE_WIDTH];
            __shared__ float N[MULT_TILE_WIDTH][MULT_TILE_WIDTH];
            
            // Need to linearize the block matrix
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // Row and Col of C's element that is being worked on
            int row = by*MULT_TILE_WIDTH + ty;
            int col = bx*MULT_TILE_WIDTH + tx;

            #define y3d(i2, i1, i0) y[(i2) * (numWRows*numXCols) + (i1) * (numXCols) + i0]

            float sum = 0.0;
            // The loop will be over a linearized tile index
            for (int tile_x = 0; tile_x < ceil(numWCols/(float)MULT_TILE_WIDTH); tile_x++) {

                // First load the values of M and N that this thread is reponsible for
                if ((row < numWRows) && (tile_x*MULT_TILE_WIDTH + tx) < numWCols) {
                    M[ty][tx] = w[row*numWCols + tile_x*MULT_TILE_WIDTH + tx];      
                } else {
                    M[ty][tx] = 0.0;
                }
                //__syncthreads();

                if((col < numXCols) && (tile_x*MULT_TILE_WIDTH + ty) < numXRows) {
                    N[ty][tx] = x_unrolled[(tile_x*MULT_TILE_WIDTH + ty)*numXCols + col];
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
            if (row < numWRows && col < numXCols) {
                y[row*numXCols + col] = sum;
            }

            #undef y3d
        }


        /*
           Host launchers
        
           - shared_conv_launcher
           - fusion_launcher
           - unroll_matmult_launcher
           -register_tiled_fusion_launcher
         */

        void shared_conv_launcher(float *y, const float * x, const float *w,  int B, int C, int M, int H,  int W,  int K) {

            const int W_out = W-K+1;  // output columns
            const int H_out = H-K+1;  // output rows

            const int W_grid = ceil((W_out)/TILE_WIDTH)+1;
            const int H_grid = ceil((H_out)/TILE_WIDTH)+1;    
            const int Z = W_grid*H_grid;

            dim3 gridDim(B, M, Z);
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

            size_t shmem_size = sizeof(float)*(TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+(K*K)*sizeof(float);
            
            shared_conv_kernel<<<gridDim, blockDim, shmem_size>>>(y, x, w, B, C, M, H, W, K);

        }

        void fusion_launcher(float *y, const float * x, const float *w, int B, int C, int M, int H,  int W,  int K) {
            int output_rows = M;
            int output_cols = (H-K+1)*(W-K+1);
            
            // Normal fusion kernel
            dim3 dimGrid(ceil(output_cols/(1.0*FUSION_TILE)), ceil(output_rows/(1.0*FUSION_TILE)), B);
            dim3 dimBlock(FUSION_TILE, FUSION_TILE, 1);
            fusion_kernel<<<dimGrid, dimBlock>>>(y, x, w, C, M, H, W, K);
        }

        void register_tiled_fusion_launcher(float *y, const float * x, const float *w, int B, int C, int M, int H,  int W,  int K) {
            int output_rows = M;
            int output_cols = (H-K+1)*(W-K+1);

            dim3 registerTileGrid(ceil(output_cols/(1.0*TILE_WIDTH_X)), ceil(output_rows/(1.0*TILE_WIDTH_W)), B);
            dim3 registerTileBlock(TILE_WIDTH_W,1,1);
            register_tiled_fusion_kernel<<<registerTileGrid, registerTileBlock>>>(y, x, w, C, M, H, W, K);
        }

        void unroll_matmult_launcher(float *y, const float * x, const float *w, int B, int C, int M, int H,  int W,  int K) {
            int H_out = H-K+1;
            int W_out = W-K+1;
            int num_threads = C*H_out*W_out;

            float * x_unrolled;
            
            cudaMalloc((void**)&x_unrolled, sizeof(float)*C*K*K*W_out*H_out);

            dim3 unrollGrid(ceil(num_threads/(1.0*MAX_THREADS)),1, 1);
            dim3 unrollBlock(MAX_THREADS, 1, 1);

            dim3 matmultGrid(ceil((H_out*W_out)/(1.0*MULT_TILE_WIDTH)), ceil(M/(1.0*MULT_TILE_WIDTH)), 1);
            dim3 matmultBlock(MULT_TILE_WIDTH, MULT_TILE_WIDTH, 1);

            for (int b = 0; b < B; b++) {
                unroll_kernel<<<unrollGrid, unrollBlock>>>(x, x_unrolled, b, C, H, W, K);
                cudaDeviceSynchronize();
                matmult_kernel<<<matmultGrid, matmultBlock>>>(w, x_unrolled, y + b*M*W_out*H_out ,M, C*K*K, C*K*K, W_out*H_out);
            }  
            
            cudaFree(x_unrolled);    
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

            // Call the launchers for various methods

            //shared_conv_launcher(y.dptr_, x.dptr_, w.dptr_, B, C, M, H, W, K);
            // unroll_matmult_launcher(y.dptr_, x.dptr_, w.dptr_, B, C, M, H, W, K);
            //fusion_launcher(y.dptr_, x.dptr_, w.dptr_, B, C, M, H, W, K);
            register_tiled_fusion_launcher(y.dptr_, x.dptr_, w.dptr_, B, C, M, H, W, K);


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
