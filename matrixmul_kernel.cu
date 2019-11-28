#define MAX_THREADS 1024
#define MULT_TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
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

void matrixMult(int M, int C, int K, int b, int H_out, int W_out, float *w, float *x, float *y){ 
    int A_rows = M;
    int A_columns = C * K * K;
    int B_columns = H_out * W_out;

    dim3 dimGrid(ceil(B_columns/(1.0*MULT_TILE_WIDTH)), ceil(A_rows/(1.0*MULT_TILE_WIDTH)), 1);
    dim3 dimBlock(MULT_TILE_WIDTH, MULT_TILE_WIDTH, 1);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(w, x, y+b*M*B_columns, A_rows, A_columns, A_columns, B_columns, A_rows, B_columns);
}

void matrixMult_launcher(int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols, float *w, float *x, float *y){ 
    int A_rows = a_rows;
    int A_columns = a_cols;
    int B_columns = b_cols;

    dim3 dimGrid(ceil(B_columns/(1.0*MULT_TILE_WIDTH)), ceil(A_rows/(1.0*MULT_TILE_WIDTH)), 1);
    dim3 dimBlock(MULT_TILE_WIDTH, MULT_TILE_WIDTH, 1);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(w, x, y, A_rows, A_columns, A_columns, B_columns, A_rows, B_columns);
}
