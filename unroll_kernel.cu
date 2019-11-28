__global__ void unroll_inputs(const int C, const int H, const int W, const int K, const int b, const float * x, float * x_unrolled) {
    int c, s, h_out, w_out, h_unroll, w_base;

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    int t = threadIdx.x + MAX_THREADS*blockIdx.x;
    
    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll; 
        h_out = s / W_out;
        w_out = s % W_out;
        h_unroll = h_out * W_out + w_out;
        w_base = c * K * K;

        #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
        #define x2d(i1, i0) x_unrolled[(i1) * (H_out*W_out) + (i0)]
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int w_unroll = w_base + p*K + q;
                x2d(w_unroll, h_unroll) = x4d(b, c, h_out+p, w_out+q);
            }
        }
    }
 
    #undef x4d
    #undef x2d
}

void unroll_input(int C, int H, int W, int K, int b, float * x, float * x_unrolled){
    int H_out = H-K+1;
    int W_out = W-K+1;
    int num_threads = C*H_out*W_out;
    dim3 unrollGrid(ceil(num_threads/(1.0*MAX_THREADS)),1, 1);
    dim3 unrollBlock(MAX_THREADS, 1, 1);			
    unroll_inputs<<<unrollGrid, unrollBlock>>>(C, H, W, K, b, x, x_unrolled);
}
