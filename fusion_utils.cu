#include "cpu_utils.cu"
__global__ void forward_pass(float * out, float * in, float * weights, const int B, const int M, const int C, const int H, const int W, const int K)
{
     const int H_out = H - K + 1;
     const int W_out = W - K + 1;
     
     extern __shared__ float shared_mem[];
     
     int X_tile_size = TILE_WIDTH + K - 1;
     
     int W_grid = (int)((W_out-.5)/TILE_WIDTH+0.5);
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int bz = blockIdx.z;
     int tx = threadIdx.x;
     int ty = threadIdx.y;

     float * x_shared = (float *) &shared_mem[0];
     float * k_shared = (float *) &shared_mem[X_tile_size*X_tile_size];
     int th = ty;
     int tw = tx;
     
     int w_base = (bz % W_grid) * TILE_WIDTH;
     int h_base = (bz / W_grid) * TILE_WIDTH;

     int w = w_base + tw;
     int h = h_base + th;
     #define out4d(i3, i2, i1, i0) out[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
     #define in4d(i3, i2, i1, i0) in[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
     #define mask4d(i3, i2, i1, i0) weights[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

     float acc = 0;

     for(int c = 0; c < C; c++) {
	 //load mask elements into shared memory
	 __syncthreads();
	 if ((th < K) && (tw < K)) {
	     k_shared[th*K+tw] = mask4d(by, c, th, tw);
	 }


	 //load input elements into shared
	 //memory
	 for (int i = h; i < h_base + X_tile_size; i+= TILE_WIDTH) {
	     __syncthreads();
	     for (int j = w; j < w_base + X_tile_size; j += TILE_WIDTH) {
		 if (i < H && j < W) {
		     x_shared[(i-h_base)*X_tile_size+(j-w_base)] = in4d(bx, c, i, j);
		 } else {
		     x_shared[(i-h_base)*X_tile_size+(j-w_base)] = 0.0f;
		 }
	     }
	 }


	 //perform convolution
	 for(int p = 0; p < K; p++) {
	     for (int q = 0; q < K; q++) {
		 acc += x_shared[(th+p)*X_tile_size+(tw+q)]* k_shared[p*K+q];
		  //acc += x_shared[(th+p)*X_tile_size+(tw+q)] * mask4d(by, c, p, q);
	     }
	 }
	 __syncthreads();
     }


     //only threads with valid indices contribute to output vector
     if (h < H_out && w < W_out) {
	  out4d(bx, by, h, w) = acc;
	  //out4d(bx, by, h, w) = 0.05;
      }
     #undef out4d
     #undef in4d
     #undef mask4d
}
void forward_launcher(vector_t * out, vector_t * in, vector_t * mask)
{
    int B = in->hyper;
    int M = out->depth;
    int C = in->depth;
    int H = in->rows;
    int W = in->cols;
    int K = mask->depth;
    
    int W_out = W - K + 1;
    int H_out = H - K + 1;

    int W_grid = ceil((W_out)/TILE_WIDTH)+1;
    int H_grid = ceil((H_out)/TILE_WIDTH)+1;

    int Z = W_grid * H_grid;
    
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    size_t shmem_size = sizeof(float) * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + (K * K) * sizeof(float);
    forward_pass<<<gridDim, blockDim, shmem_size>>>(out->data, in->data, mask->data, B, M, C, H, W, K);
}
int main(void) 
{
    srand(time(0));
    vector_t * h_vector = alloc_vector(A_ROWS, A_COLS, A_DEPTH, A_HYPER, IDENTITY);
    vector_t * h_weights = alloc_vector(K_ROWS, K_COLS, A_DEPTH, K_HYPER, IDENTITY);
    vector_t * h_output = alloc_vector(A_ROWS - K_ROWS + 1, A_COLS - K_COLS + 1, A_DEPTH, K_HYPER,  ZEROES);
    vector_t * cpu_output = alloc_vector(A_ROWS - K_ROWS + 1, A_COLS - K_COLS + 1, A_DEPTH, K_HYPER, ZEROES);
    
    float * d_vector;
    float * d_weights;
    float * d_output;
    float * h_vector_out = (h_output->data);
    cudaMalloc((float**)&d_vector, h_vector->size);
    cudaMalloc((float**)&d_weights, h_weights->size);
    cudaMalloc((float**)&d_output, h_output->size);
    cudaMemcpy(d_vector, h_vector->data, h_vector->size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights->data, h_weights->size, cudaMemcpyHostToDevice);
    
    int B = h_vector->hyper;
    int M = h_output->depth;
    int C = h_vector->depth;
    int H = h_vector->rows;
    int W = h_vector->cols;
    int K = h_weights->depth;

    int W_out = W - K + 1;
    int H_out = H - K + 1;

    int W_grid = ceil((W_out)/TILE_WIDTH)+1;
    int H_grid = ceil((H_out)/TILE_WIDTH)+1;

    int Z = W_grid * H_grid;

    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    size_t shmem_size = sizeof(float) * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + (K * K) * sizeof(float);

    printf("Vector A: \n");
    print_vector(h_vector);
    printf("\nVector B (weights): \n");
    print_vector(h_weights);

    printf("Launching kernel...\n");
    
    printf("Initializing convolution with params: \nB\t: %d\nM\t: %d\nC\t: %d\nH\t: %d\nW\t: %d\nK\t: %d\n", B, M, C, H, W, K);
    printf("X_tile_width: %d\n", TILE_WIDTH+K-1);
    
    
    printf("Launching Thread Grid with: \ngridDim: x\t: %d\t y\t: %d\tz\t: %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("blockDim: x\t: %d\ty\t: %d\tz\t: %d\n", blockDim.x, blockDim.y, blockDim.z);			    


    //forward_launcher(dv_output, dv_vector, dv_weights);
    forward_pass<<<gridDim, blockDim, shmem_size>>>(d_output, d_vector, d_weights, B, M, C, H, W, K);
    cudaMemcpy(h_vector_out, d_output, h_output->size, cudaMemcpyDeviceToHost);
    h_output->data = h_vector_out;
    printf("Output Vector: \n");
    print_vector(h_output);
    printf("\n[CPU] Output Vector: \n");
    forward_pass_cpu(cpu_output, h_vector, h_weights);
    print_vector(cpu_output);

    printf("\n\nChecking Results...\n");
    tuple4d_t * err = (tuple4d_t*) malloc(sizeof(tuple4d_t));
    tuple4d_t * ref = err;
    int status = vector_equal(h_output, cpu_output, &ref);
    if (status){
	while (err != NULL) {
	    if (err->x >= 0 && err->x < h_output->cols && err->y >= 0 && err->y < h_output->rows && err->h >= 0 && err->h < h_output->hyper && err->w >= 0 && err->w < h_output->depth){
		printf("Error at [%d][%d][%d][%d]\n", err->h, err->w, err->x, err->y);
	    }
	    err = err->next;
	}
    } else {
	printf("Results are consistent.\n");
    }
    return 0;
}
