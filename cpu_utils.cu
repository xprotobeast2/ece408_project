#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cpu_utils.h"
#include "matrixmul_kernel.cu"
#include "unroll_kernel.cu"

int matrix_equal(matrix_t *A, matrix_t *B, tuple_t ** err) 
{
    if (!A || !A->data || !B || !B->data)
    	return -1;
    if ( !(A->rows == B->rows && A->cols == B->cols))
        return -1;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
	    int offset = i * A->cols + j;
	    if (A->data[offset] != B->data[offset]) {
	        if (!(*err)) {
		    *err = (tuple_t*) malloc(sizeof(tuple_t));
		    (*err)->x = j;
		    (*err)->y = i;
		} else {
		    tuple_t * error = (tuple_t*) malloc(sizeof(tuple_t));
		    error->x = j;
		    error->y = i;
		    error->next = *err;
		    *err = error;
		    
		}
		
	    }
	}
    }
    return 0;
}

int vector_equal(vector_t *A, vector_t *B, tuple4d_t ** err) 
{
    if (!A || !A->data || !B || !B->data)
    	return -1;
    if ( !(A->rows == B->rows && A->cols == B->cols && A->depth == B->depth && A->hyper == B->hyper))
        return -1;
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
	    for (int k = 0; k < A->depth; k++) {
	        for (int m = 0; m < A->hyper; m++) {
	             int offset = (m) * (A->rows * A->cols * A->depth) + (k) * (A->rows * A->cols) + (i) * (A->cols) +j;
	             if (A->data[offset] != B->data[offset]) {
	                 if (!(*err)) {
		              *err = (tuple4d_t*) malloc(sizeof(tuple4d_t));
		   	      (*err)->x = j;
		    	      (*err)->y = i;
		    	      (*err)->w = k;
		              (*err)->h = m;
		         } else {
		    	      tuple4d_t * error = (tuple4d_t*) malloc(sizeof(tuple4d_t));
		   	      error->x = j;
		              error->y = i;
		   	      error->w = k;
		              error->h = m;
		              error->next = *err;
		              *err = error;
			 }
		     }
		 }
	    }
       }
   }

    return 0;
}
void forward_pass_cpu(vector_t * out_vec, vector_t * in_vec, vector_t * mask)
{
    if (!out_vec || !out_vec->data || !in_vec || !in_vec->data || !mask || !mask->data)
        return;
    int B = in_vec->hyper; //batch index
    int M = out_vec->depth; //output number of channels
    int C = in_vec->depth; //input number of channels
    int H = in_vec->rows; //input rows
    int W = in_vec->cols; //input cols
    int K = mask->depth; //mask width (assuming square mask)
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    #define out4d(i3, i2, i1, i0) out_vec->data[ (i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) +(i0)]
    #define in4d(i3, i2, i1, i0) in_vec->data[ (i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) +(i0)]
    #define mask4d(i3, i2, i1, i0) mask->data[ (i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) +(i0)]
    
    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
	    for (int h = 0; h < H - K + 1; ++h) {
	        for (int w = 0; w < W - K + 1; ++w) {
		   out4d(b, m, h, w) = 0.0f;

		   for (int c = 0; c < C; ++c) {
		       for (int p = 0; p < K; ++p) {
		           for (int q = 0; q < K; ++q) {
			       out4d(b, m, h, w) += in4d(b, c, h+p, w+q) * mask4d(m, c, p, q);
			   }
		       }
		   }
	       }
	   }
        }
    }
    #undef mask4d
    #undef in4d
    #undef out4d 
}

void print_vector(vector_t *in)
{
    if (!in || !in->data)
        return;
    #define v4d(i3, i2, i1, i0) in->data[ (i3) * (in->rows * in->cols * in->depth) + (i2) * (in->rows * in->cols) + (i1) * (in->cols) +(i0)]
    for (int i = 0; i < in->rows; i++){
        for (int j = 0; j < in->cols; j++) {
	    for (int k = 0; k < in->depth; k++) {
	        for (int m = 0; m < in->hyper; m++){
		    //printf("vec[%d][%d][%d][%d]: ", m, k, i, j);
		    printf("%.1f \n", v4d(m, k, i, j));
		}
	    }
	}
	printf("\n");
    }
    printf("\n");
    #undef v4d
}

vector_t* alloc_vector(int rows, int cols, int depth, int hyper, matrix_init_t mode)
{	
	vector_t* vec = (vector_t*) malloc(sizeof(vector_t));

	vec->rows = rows;
	vec->cols = cols;
	vec->depth = depth;
	vec->hyper = hyper;
	vec->size = vec->rows * vec->cols * vec->depth * vec->hyper * sizeof(float);
	vec->data = (float*) malloc(vec->size);

	int i, j, w, h;
	//int r;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
		    for (w = 0; w < depth; w++) {
		        for (h = 0; h < hyper; h++) {
		            #define v4d(i3, i2, i1, i0) vec->data[ (i3) * (vec->rows * vec->cols * vec->depth) + (i2)*(vec->rows * vec->cols) + (i1) * (vec->cols) +(i0)]
			switch (mode) {
				case IDENTITY:
					v4d(h, w, i, j) = (i == j) ? 1.0f : 0.0f;
					break;
				case RAND:
				        //r = get_rand_val();
					//printf("Assigned: vec[%d][%d][%d][%d] = %d\n", h, w, i, j, r);
					v4d(h, w, i, j) = get_rand_val();
					//v4d(h, w, i, j) = r;
					break;
				case WEIGHTS:
					v4d(h, w, i, j) = (w % 2 == 0) ? 0.5 : 1;
					break;
				case ZEROES:
				default:
					v4d(h, w, i, j) = 0.0f;
					break;
			}
		    }
	        }
	    }
	}

	#undef v4d
	return vec;
}
void print_matrix(matrix_t *in) 
{
    if (in->data == NULL || in == NULL)
		return;
	for (int i = 0; i < in->rows; i++) {
		for (int j = 0; j < in->cols; j++) {
			printf("%.1f ", in->data[i * in->cols + j]);
		}
		printf("\n");
	}
}

int get_rand_val()
{
	return rand() % Q + 1;
}

matrix_t* alloc_matrix(int rows, int cols, matrix_init_t mode)
{	
	matrix_t* mat = (matrix_t*) malloc(sizeof(matrix_t));

	mat->rows = rows;
	mat->cols = cols;
	mat->size = rows * cols * sizeof(float);
	mat->data = (float*) malloc(mat->size);

	int i, j, offset;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			offset = j + i * mat->cols;
			switch (mode) {
				case IDENTITY:
					mat->data[offset] = (i == j) ? 1.0f : 0.0f;
					break;
				case RAND:
					mat->data[offset] = get_rand_val();
					break;
				case ZEROES:
				default:
					mat->data[offset] = 0.0f;
					break;
			}
		}
	}

	return mat;
}

void matrix_mult_cpu(matrix_t * A, matrix_t * B, matrix_t * C)
{
	if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL)
		return;
	int i, j, k;
	for (i = 0; i < A->rows; i++) {
		for (j = 0; j < C->cols; j++) {
			C->data[i* C->cols + j] = 0.0f;
			for (k = 0; k < B->rows; k++) {
				C->data[i * C->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
			}
		}
	}
}
/*
int main(void) 
{
	srand(time(0));
	matrix_t * h_A = alloc_matrix(A_ROWS, A_COLS, RAND);
	matrix_t * h_B = alloc_matrix(B_ROWS, B_COLS, RAND);
	matrix_t * h_C = alloc_matrix(A_ROWS, B_COLS, ZEROES);
	matrix_t * cpu_result = alloc_matrix(A_ROWS, B_COLS, ZEROES);
	float * d_A;
	float * d_B;
	float * d_C;
	cudaMalloc((float**)&d_A, h_A->size);
	cudaMalloc((float**)&d_B, h_B->size);
	cudaMalloc((float**)&d_C, h_C->size);
	cudaMemcpy(d_A, h_A->data, h_A->size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B->data, h_B->size, cudaMemcpyHostToDevice);

        printf("Matrix A: \n");
	print_matrix(h_A);
	printf("\nMatrix B: \n");
	print_matrix(h_B);
	
	printf("Launching kernel...\n");
	matrixMult_launcher(h_A->rows, h_A->cols, h_B->rows, h_B->cols, h_C->rows, h_C->cols, d_A, d_B, d_C);
	cudaMemcpy(h_C->data, d_C, h_C->size, cudaMemcpyDeviceToHost);
	printf("Matrix C: \n");
	print_matrix(h_C);
	printf("\n[CPU] Matrix C: \n");
	matrix_mult_cpu(h_A, h_B, cpu_result);
	print_matrix(cpu_result);
	
	printf("\n\nChecking Results...\n");
	tuple_t * err = (tuple_t*) malloc(sizeof(tuple_t));
	tuple_t * ref = err;
	int status = matrix_equal(h_C, cpu_result, &ref);
	if (status){
	    while (err != NULL) {
	        if (err->x >= 0 && err->x < h_C->cols && err->y >= 0 && err->y < h_C->rows){
	            printf("Error at [%d][%d]\n", err->x, err->y);
		}
		err = err->next;
	    }
	} else {
	    printf("Results are consistent.\n");
	}
	printf("\nGenerating 4D vector representation...\n");
	vector_t * h_V = alloc_vector(A_ROWS, A_COLS, A_DEPTH, A_HYPER, RAND);
	printf("Vector V:\n");
	print_vector(h_V);
	return 0;
}*/
