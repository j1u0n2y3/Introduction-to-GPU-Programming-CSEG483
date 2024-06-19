#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "measure_host_time.h"
#include "mma.h"
#include "cublas_v2.h"
using namespace nvcuda;

#define VAL_IDX 2 /* 0, 1, 2 */
#define TS 32
#define WPT 8
#define RTS (TS / WPT)

int _Arow[3] = {1024, 1 << 14, 1021},
    _Acol[3] = {2048, 1 << 12, 2039},
    _Brow[3] = {_Acol[0], _Acol[1], _Acol[2]},
    _Bcol[3] = {3200, 1 << 13, 3203},
    _Crow[3] = {_Arow[0], _Arow[1], _Arow[2]},
    _Ccol[3] = {_Bcol[0], _Bcol[1], _Bcol[2]};

void init_flt_data(float arr[], int n)
{
    std::default_random_engine gen(20211584);
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for (int i = 0; i < n; i++)
        arr[i] = fran(gen);
}

void init_hf_data(half dst[], float src[], int n)
{
    for (int i = 0; i < n; i++)
        dst[i] = (half)src[i];
}

template <typename T1, typename T2>
void compare_two_matrices(T1 *ori, T2 *cmp, int Arow, int Acol, float *ave_rel_error, float *max_rel_error)
{
    double rel_error_sum = 0.0;
    float rel_error_max = 0.0f;
    for (int k = 0; k < Arow * Acol; k++)
    {
        float rel_error = fabsf(((float)cmp[k] - (float)ori[k]) / (float)ori[k]);
        rel_error_sum += rel_error;
        if (rel_error > rel_error_max)
            rel_error_max = rel_error;
    }
    *ave_rel_error = rel_error_sum / (Arow * Acol);
    *max_rel_error = rel_error_max;
}

void MM_HOST(double *C, float *A, float *B, int Ay, int Ax, int Bx)
{
    for (int i = 0; i < Ay; i++)
    {
        for (int k = 0; k < Ax; k++)
        {
            double Aik = (double)A[i * Ax + k];
            for (int j = 0; j < Bx; j++)
            {
                C[i * Bx + j] += Aik * (double)B[k * Bx + j];
            }
        }
    }
}

__global__ void MM_DEVICE_GM(float *__restrict C, const float *__restrict A, const float *__restrict B,
                             int Ay, int Ax, int Bx)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (ty >= Ay || tx >= Bx)
        return;

    float ctmp = 0.0f;
#pragma unroll 16
    for (int k = 0; k < Ax; k++)
    {
        ctmp += A[ty * Ax + k] * B[k * Bx + tx];
    }
    C[ty * Bx + tx] = ctmp;
}

__global__ void MM_DEVICE_SM(float *__restrict C, float *__restrict A, float *__restrict B, int Ay, int Ax, int Bx)
{
    __shared__ float Atile[TS][TS]; // tile in A eg [16][16]
    __shared__ float Btile[TS][TS]; // tile in B eg [16][16]

    int tx = threadIdx.x;              // tile col index j
    int ty = threadIdx.y;              // tile row index i
    int ocx = blockDim.x * blockIdx.x; // tile x origin in C (all threads)
    int ocy = blockDim.y * blockIdx.y; // tile y origin in C (all threads)

    int ax = tx;       // j or x in first tile on A
    int ay = ocy + ty; // i or y in first tile on A and C
    int bx = ocx + tx; // j or x in first tile on B and C
    int by = ty;       // i or y in first tile on B

    float csum = 0.0f;
#pragma unroll 16
    for (int t = 0; t < Ax / TS; t++)
    {
        Atile[ty][tx] = A[ay * Ax + ax]; // copy A tile to shared mem
        Btile[ty][tx] = B[by * Bx + bx]; // copy B tile to shared mem
        __syncthreads();
        for (int k = 0; k < TS; k++)
            csum += Atile[ty][k] * Btile[k][tx];
        __syncthreads();
        ax += TS; // step A tiles along rows of A
        by += TS; // step B tiles down  cols of B
    }
    C[ay * Bx + bx] = csum; // store complete result
}

__global__ void MM_DEVICE_SM_MWPT(float *__restrict C, const float *__restrict A, const float *__restrict B,
                                  int Ay, int Ax, int Bx)
{
    __shared__ float Atile[TS][TS];
    __shared__ float Btile[TS][TS];
    float accum[WPT] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y * WPT;

    int Row = by + ty; // Row index for C
    int Col = bx + tx; // Column index for C

    for (int t = 0; t < (Ax + TS - 1) / TS; t++)
    {
        // Load one tile of A and B into shared memory
        for (int w = 0; w < WPT; w++)
        {
            int Arow = Row + w * RTS;
            int Acol = t * TS + tx;
            int Brow = t * TS + ty + w * RTS;
            int Bcol = Col;
            if (Arow < Ay && Acol < Ax)
                Atile[ty + w * RTS][tx] = A[Arow * Ax + Acol];
            else
                Atile[ty + w * RTS][tx] = 0.0f;
            if (Brow < Ax && Bcol < Bx)
                Btile[ty + w * RTS][tx] = B[Brow * Bx + Bcol];
            else
                Btile[ty + w * RTS][tx] = 0.0f;
        }
        __syncthreads();

        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++)
        {
            for (int w = 0; w < WPT; w++)
            {
                accum[w] += Atile[ty + w * RTS][k] * Btile[k][tx];
            }
        }
        __syncthreads();
    }

    // Store the final results in C
    for (int w = 0; w < WPT; w++)
    {
        int Cindex = (Row + w * RTS) * Bx + Col;
        if ((Row + w * RTS) < Ay && Col < Bx)
            C[Cindex] = accum[w];
    }
}

__global__ void MM_DEVICE_TC_GM(float *__restrict C, const half *__restrict A, const half *__restrict B,
                                int Ay, int Ax, int Bx)
{
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize; // warp rank in grid

    int cx = warp % (Bx / 16); // (x,y) location of active tile
    int cy = warp / (Bx / 16); // for current warp in C matrix

    int Atile_pos = cy * 16 * Ax; // start x (row) for first A tile
    int Btile_pos = cx * 16;      // start y (col) for first B tile

    // Declare the fragments as 16 x 16 tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // A
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; // B
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;              // C
    wmma::fill_fragment(c_frag, 0.0f);                                        // set C = 0

    for (int k = 0; k < Ax / 16; k++)
    {                                                      // accumulate su, of row*column for C tile
        wmma::load_matrix_sync(a_frag, &A[Atile_pos], Ax); // load A as 16x16 tile
        wmma::load_matrix_sync(b_frag, &B[Btile_pos], Bx); // load B as 16x16 tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);    // C = A*B + C
        Atile_pos += 16;                                   // step along row of A
        Btile_pos += 16 * Bx;                              // step down column of B
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}

__global__ void MM_DEVICE_TC_SM(float *__restrict C, const half *__restrict A, const half *__restrict B,
                                int Ay, int Ax, int Bx)
{
    __shared__ half as[256];
    __shared__ half bs[8][256];

    if (blockDim.x != 256)
        return; // force 256 threads per block

    // Find row tile and 8 col tiles for this thread block
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;

    int cx = warp % (Bx / 16);
    int cy = warp / (Bx / 16);

    int Atile_pos = cy * 16 * Ax; // A starts 1 left row at cy
    int Btile_pos = cx * 16;      // B starts 8 top cols at cx

    int wb = threadIdx.x / 32;  // warp rank in block  in [0,255]
    int trw = threadIdx.x % 32; // thread rank in warp
    int txw = trw % 16;         // thread x in warp    in [0,15]
    int tyw = trw / 16;         // thread y in warp    in [0, 1]

    int idx = threadIdx.x % 16; // assign 256 threads to cover
    int idy = threadIdx.x / 16; // 16 x 16 x-y values in tile

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // A
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; // B
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;              // C
    wmma::fill_fragment(c_frag, 0.0f);                                        // set C = 0

    for (int k = 0; k < Ax / 16; k++)
    {
        as[idy * 16 + idx] = A[Atile_pos + idy * Ax + idx]; // 256 threads used here
        __syncthreads();                                    // 32 threads fill tile in 8 passes
        for (int p = 0; p < 8; p++)
            bs[wb][p * 32 + tyw * 16 + txw] = B[p * 2 * Bx + Btile_pos + tyw * Bx + txw];
        __syncwarp();
        wmma::load_matrix_sync(a_frag, &as[0], 16);     // load A as 16x16 tile
        wmma::load_matrix_sync(b_frag, &bs[wb][0], 16); // load B as 16x16 tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag); // C = A*B + C
        Atile_pos += 16;                                // move along A row
        Btile_pos += 16 * Bx;                           // move down B cols
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}

void MM_DEVICE_CUBLAS(float *C, float *A, float *B, int Ay, int Ax, int Bx)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                Bx, Ay, Ax,
                &alpha, // 1
                B, Bx,
                A, Ax,
                &beta, // 0
                C, Bx);
}

int main(int argc, char *argv[])
{
    int Arow = _Arow[VAL_IDX],
        Acol = _Acol[VAL_IDX],
        Brow = _Brow[VAL_IDX],
        Bcol = _Bcol[VAL_IDX],
        Crow = _Crow[VAL_IDX],
        Ccol = _Ccol[VAL_IDX];

    /* host mem */
    float *h_flt_A = new float[Arow * Acol],
          *h_flt_B = new float[Brow * Bcol];
    init_flt_data(h_flt_A, Arow * Acol);
    init_flt_data(h_flt_B, Brow * Bcol);
    half *h_hf_A = new half[Arow * Acol],
         *h_hf_B = new half[Brow * Bcol];
    init_hf_data(h_hf_A, h_flt_A, Arow * Acol);
    init_hf_data(h_hf_B, h_flt_B, Brow * Bcol);
    double *h_db_C = new double[Crow * Ccol]{};
    float *h_res_C = new float[Crow * Ccol];

    /* device mem */
    float *d_flt_A, *d_flt_B;
    cudaMalloc((void **)&d_flt_A, Arow * Acol * sizeof(float));
    cudaMalloc((void **)&d_flt_B, Brow * Bcol * sizeof(float));
    cudaMemcpy(d_flt_A, h_flt_A, Arow * Acol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flt_B, h_flt_B, Brow * Bcol * sizeof(float), cudaMemcpyHostToDevice);
    half *d_hf_A, *d_hf_B;
    cudaMalloc((void **)&d_hf_A, Arow * Acol * sizeof(half));
    cudaMalloc((void **)&d_hf_B, Brow * Bcol * sizeof(half));
    cudaMemcpy(d_hf_A, h_hf_A, Arow * Acol * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hf_B, h_hf_B, Brow * Bcol * sizeof(half), cudaMemcpyHostToDevice);
    float *d_flt_C;
    cudaMalloc((void **)&d_flt_C, Crow * Ccol * sizeof(float));

    /* [1] */
    float ave_rel_error, max_rel_error;
    float host_time_flt;
    CHECK_TIME_START(_start, _freq);
    MM_HOST(h_db_C, h_flt_A, h_flt_B, Arow, Acol, Bcol);
    CHECK_TIME_END(_start, _end, _freq, host_time_flt);
    fprintf(stdout, "[1] Host time(double) = %f(ms) -----------------------------\n", host_time_flt);

    compare_two_matrices<float, half>(h_flt_A, h_hf_A, Arow, Acol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_A_flt and h_A_hf] average = %f, maximum = %f\n", ave_rel_error, max_rel_error);
    compare_two_matrices<float, half>(h_flt_B, h_hf_B, Brow, Bcol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_B_flt and h_B_hf] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [2] */
    dim3 threads = {32, 16, 1};
    dim3 blocks = {(Bcol + threads.x - 1) / threads.x, (Arow + threads.y - 1) / threads.y, 1};

    MM_DEVICE_GM<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_CC_flt;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_GM<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_flt);
    fprintf(stdout, "[2] GPU time(CUDA Cores/float) = %f(ms) -----------------------------\n", gpu_time_CC_flt);

    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [3] */
    threads = {32, 32, 1};
    blocks = {(Bcol + threads.x - 1) / threads.x, (Arow + threads.y - 1) / threads.y, 1};

    MM_DEVICE_SM<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_CC_flt_shared;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_SM<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_flt_shared);
    fprintf(stdout, "[3] GPU time(CUDA Cores/float/shared memory) = %f(ms) -----------------------------\n", gpu_time_CC_flt_shared);

    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [4] */
    threads = {TS, RTS, 1};
    blocks = {(Bcol + threads.x - 1) / threads.x, (Arow + threads.y * WPT - 1) / (threads.y * WPT), 1};

    MM_DEVICE_SM_MWPT<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_CC_flt_shared_MWpT;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_SM_MWPT<<<blocks, threads>>>(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_flt_shared_MWpT);
    fprintf(stdout, "[4] GPU time(CUDA Cores/float/shared memory/More-Work-per-Thread) = %f(ms) -----------------------------\n", gpu_time_CC_flt_shared_MWpT);

    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(CUDA Cores/float)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [5] */
    int blocksT = Arow * Bcol / (8 * 256);
    MM_DEVICE_TC_GM<<<blocksT, 256>>>(d_flt_C, d_hf_A, d_hf_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_TC_hf;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_TC_GM<<<blocksT, 256>>>(d_flt_C, d_hf_A, d_hf_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_TC_hf);
    fprintf(stdout, "[5] GPU time(Tensor Cores/half) = %f(ms) -----------------------------\n", gpu_time_TC_hf);

    ave_rel_error = 999;
    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [6] */
    blocksT = Arow * Bcol / (8 * 256);
    MM_DEVICE_TC_SM<<<blocksT, 256>>>(d_flt_C, d_hf_A, d_hf_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_TC_hf_shared;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_TC_SM<<<blocksT, 256>>>(d_flt_C, d_hf_A, d_hf_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_TC_hf_shared);
    fprintf(stdout, "[6] GPU time(Tensor Cores/half/shared memory) = %f(ms) -----------------------------\n", gpu_time_TC_hf_shared);

    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(Tensor Cores/half/shared memory)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);

    /* [7] */
    MM_DEVICE_CUBLAS(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol); // dummy run
    cudaDeviceSynchronize();

    float gpu_time_cB_flt;
    CHECK_TIME_START(_start, _freq);
    MM_DEVICE_CUBLAS(d_flt_C, d_flt_A, d_flt_B, Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_cB_flt);
    fprintf(stdout, "[7] GPU time(cuBlas/float) = %f(ms) -----------------------------\n", gpu_time_cB_flt);

    cudaMemcpy(h_res_C, d_flt_C, Crow * Ccol * sizeof(float), cudaMemcpyDeviceToHost);
    compare_two_matrices<double, float>(h_db_C, h_res_C, Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors(cuBlas/float)] average = %f, maximum = %f\n\n", ave_rel_error, max_rel_error);
	
	return 0;
}
