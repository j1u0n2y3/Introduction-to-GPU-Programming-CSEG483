// Programming in Parallel with CUDA - supporting code by Richard Ansorge.
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use.
// This code may be freely changed but please retain an acknowledgement.

// Modified for color image from example 4.9 filter9PT_2.

#include <stdio.h>
#include <random>
#include <string.h>
#include <string>
using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "measure_host_time.h"
#include "Image_IO.h"

/* Preprocessing declarations - Please read item 2. in the report carefully. ***********************/
/* 1. */
#define VIDEO_LENGTH 10
#define ENABLE_GF_CPU
/* 2. */
#define IMAGESET_LENGTH 25
#define NUM_STREAMS_M 16
/* Image file */
#define WRITE_FILE
const string INPUT_FILE_LOC = "Data\\";
const string INPUT_FILE_NAME = "Image_5_1856_1376";
const string INPUT_FILE_EXT = ".jpg";
#define INPUT_IMG ((INPUT_FILE_LOC + INPUT_FILE_NAME + INPUT_FILE_EXT).c_str())
const string OUTPUT_FILE_LOC = "C:\\usr\\S20211584\\";
const string OUTPUT_FILE_EXT = ".png";
#define OUTPUT_IMG(num) ((OUTPUT_FILE_LOC + INPUT_FILE_NAME + "_" + num + OUTPUT_FILE_EXT).c_str())
/***************************************************************************************************/

// data explicilty in constant memory must be declared at file scope
// arrays sizes must be known at compile time.
__constant__ float filter_weight[25];
#define IDX(x, y, nx, ny) (nx * (y < 0 ? 0 : (y >= ny ? ny - 1 : y)) + (x < 0 ? 0 : (x >= nx ? nx - 1 : x)))
#define FILTER_NAME GAUSSIAN_FILTER_5

void GF_CPU(const uchar4 *__restrict input_image, uchar4 *__restrict output_image, int nx, int ny)
{
    int y_offset[5], x_offset[5];
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            y_offset[0] = max(0, y - 2), y_offset[1] = max(0, y - 1), y_offset[2] = y, y_offset[3] = min(ny - 1, y + 1), y_offset[4] = min(ny - 1, y + 2);
            x_offset[0] = max(0, x - 2), x_offset[1] = max(0, x - 1), x_offset[2] = x, x_offset[3] = min(nx - 1, x + 1), x_offset[4] = min(nx - 1, x + 2);
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    uchar4 pixel_in = input_image[nx * y_offset[i] + x_offset[j]];
                    float weight = filter_5[GAUSSIAN_FILTER_5][i * 5 + j];
                    v.x += pixel_in.x * weight;
                    v.y += pixel_in.y * weight;
                    v.z += pixel_in.z * weight;
                    v.w += pixel_in.w * weight;
                }
            }
            output_image[IDX(x, y, nx, ny)] = {(unsigned char)min(255, max(0, (unsigned int)(v.x + 0.5f))),
                                               (unsigned char)min(255, max(0, (unsigned int)(v.y + 0.5f))),
                                               (unsigned char)min(255, max(0, (unsigned int)(v.z + 0.5f))),
                                               (unsigned char)min(255, max(0, (unsigned int)(v.w + 0.5f)))};
        }
    }
}

__global__ void
GF_GLOBAL(const uchar4 *__restrict input_image, uchar4 *__restrict output_image, int nx, int ny)
{
    auto idx = [&nx](int y, int x)
    { return y * nx + x; };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= nx || y >= ny)
        return;

    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int c_index = 0;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            uchar4 pixel_in = input_image[IDX(x + dx, y + dy, nx, ny)];
            float weight = filter_weight[c_index++];
            v.x += weight * pixel_in.x;
            v.y += weight * pixel_in.y;
            v.z += weight * pixel_in.z;
            v.w += weight * pixel_in.w;
        }
    }
    output_image[IDX(x, y, nx, ny)] = {(unsigned char)min(255, max(0, (unsigned int)(v.x + 0.5f))),
                                       (unsigned char)min(255, max(0, (unsigned int)(v.y + 0.5f))),
                                       (unsigned char)min(255, max(0, (unsigned int)(v.z + 0.5f))),
                                       (unsigned char)min(255, max(0, (unsigned int)(v.w + 0.5f)))};
}

__global__ void
GF_SHARED_1(const uchar4 *__restrict input_image, uchar4 *__restrict output_image, int nx, int ny)
{
    __shared__ uchar4 tile[16 + 4][16 + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int x = bx + tx, y = by + ty;
    if (x < 0 || y < 0 || x >= nx || y >= ny)
        return;

    tile[ty + 2][tx + 2] = input_image[IDX(x, y, nx, ny)];

    if (tx < 2)
        tile[ty + 2][tx] = input_image[IDX(x - 2, y, nx, ny)];
    else if (tx >= 16 - 2)
        tile[ty + 2][tx + 4] = input_image[IDX(x + 2, y, nx, ny)];

    if (ty < 2)
        tile[ty][tx + 2] = input_image[IDX(x, y - 2, nx, ny)];
    else if (ty >= 16 - 2)
        tile[ty + 4][tx + 2] = input_image[IDX(x, y + 2, nx, ny)];

    if (tx < 2 && ty < 2)
        tile[ty][tx] = input_image[IDX(x - 2, y - 2, nx, ny)];
    else if (tx < 2 && ty >= 16 - 2)
        tile[ty + 4][tx] = input_image[IDX(x - 2, y + 2, nx, ny)];
    else if (tx >= 16 - 2 && ty < 2)
        tile[ty][tx + 4] = input_image[IDX(x + 2, y - 2, nx, ny)];
    else if (tx >= 16 - 2 && ty >= 16 - 2)
        tile[ty + 4][tx + 4] = input_image[IDX(x + 2, y + 2, nx, ny)];

    __syncthreads();

    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int c_index = 0;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            uchar4 pixel_in = tile[ty + 2 + dy][tx + 2 + dx];
            float weight = filter_weight[c_index++];
            v.x += weight * pixel_in.x;
            v.y += weight * pixel_in.y;
            v.z += weight * pixel_in.z;
            v.w += weight * pixel_in.w;
        }
    }

    output_image[IDX(x, y, nx, ny)] = make_uchar4((unsigned char)min(255, max(0, int(v.x + 0.5f))),
                                                  (unsigned char)min(255, max(0, int(v.y + 0.5f))),
                                                  (unsigned char)min(255, max(0, int(v.z + 0.5f))),
                                                  (unsigned char)min(255, max(0, int(v.w + 0.5f))));
}

__global__ void
GF_SHARED_N(const uchar4 *__restrict__ input_image, uchar4 *__restrict__ output_image, int nx, int ny)
{
    __shared__ uchar4 tile[16 + 4][16 + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * (blockDim.x * 2);
    int by = blockIdx.y * (blockDim.y * 2);

    int x = bx + 2 * tx;
    int y = by + 2 * ty;

    if (x < 0 || y < 0 || x >= nx || y >= ny)
        return;

#pragma unroll 2
    for (int dy = 0; dy < 2; dy++)
    {
#pragma unroll 2
        for (int dx = 0; dx < 2; dx++)
        {
            int lx = tx * 2 + dx;
            int ly = ty * 2 + dy;
            tile[ly + 2][lx + 2] = input_image[IDX(x + dx, y + dy, nx, ny)];

            if (lx < 2)
            {
                tile[ly + 2][lx] = input_image[IDX(x + dx - 2, y + dy, nx, ny)];
            }
            else if (lx >= 16 - 2)
            {
                tile[ly + 2][lx + 4] = input_image[IDX(x + dx + 2, y + dy, nx, ny)];
            }

            if (ly < 2)
            {
                tile[ly][lx + 2] = input_image[IDX(x + dx, y + dy - 2, nx, ny)];
                if (lx < 2)
                    tile[ly][lx] = input_image[IDX(x + dx - 2, y + dy - 2, nx, ny)];
                if (lx >= 16 - 2)
                    tile[ly][lx + 4] = input_image[IDX(x + dx + 2, y + dy - 2, nx, ny)];
            }
            else if (ly >= 16 - 2)
            {
                tile[ly + 4][lx + 2] = input_image[IDX(x + dx, y + dy + 2, nx, ny)];
                if (lx < 2)
                    tile[ly + 4][lx] = input_image[IDX(x + dx - 2, y + dy + 2, nx, ny)];
                if (lx >= 16 - 2)
                    tile[ly + 4][lx + 4] = input_image[IDX(x + dx + 2, y + dy + 2, nx, ny)];
            }
        }
    }
    __syncthreads();

#pragma unroll 2
    for (int dy = 0; dy < 2; dy++)
    {
#pragma unroll 2
        for (int dx = 0; dx < 2; dx++)
        {
            int lx = tx * 2 + dx;
            int ly = ty * 2 + dy;

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int c_index = 0;
            for (int dyf = -2; dyf <= 2; dyf++)
            {
                for (int dxf = -2; dxf <= 2; dxf++)
                {
                    uchar4 pixel_in = tile[ly + 2 + dyf][lx + 2 + dxf];
                    float weight = filter_weight[c_index++];
                    v.x += weight * pixel_in.x;
                    v.y += weight * pixel_in.y;
                    v.z += weight * pixel_in.z;
                    v.w += weight * pixel_in.w;
                }
            }

            output_image[IDX(x + dx, y + dy, nx, ny)] = make_uchar4((unsigned char)min(255, max(0, int(v.x + 0.5f))),
                                                                    (unsigned char)min(255, max(0, int(v.y + 0.5f))),
                                                                    (unsigned char)min(255, max(0, int(v.z + 0.5f))),
                                                                    (unsigned char)min(255, max(0, int(v.w + 0.5f))));
        }
    }
}

__global__ void
GF_COMPARE(const uchar4 *__restrict__ input1, const uchar4 *__restrict__ input2, uchar4 *__restrict__ output, int nx, int ny)
{
    auto idx = [&nx](int y, int x)
    { return y * nx + x; };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= nx || y >= ny)
        return;

    float4 out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    uchar4 in1 = input1[idx(y, x)];
    uchar4 in2 = input2[idx(y, x)];
    out.x = abs(in1.x - in2.x);
    out.y = abs(in1.y - in2.y);
    out.z = abs(in1.z - in2.z);
    output[idx(y, x)] = {(unsigned char)min(255, max(0, (unsigned int)(out.x + 0.5f))),
                         (unsigned char)min(255, max(0, (unsigned int)(out.y + 0.5f))),
                         (unsigned char)min(255, max(0, (unsigned int)(out.z + 0.5f))),
                         (unsigned char)255};
}

void IMAGESET_DEFAULT(IO_Images *video, int video_length)
{
    uchar4 *d_input, *d_output;
    cudaMalloc(&d_input, video[0].data_bytes);
    cudaMalloc(&d_output, video[0].data_bytes);
    dim3 threads = {16, 16, 1};
    dim3 blocks = {(video[0].width + threads.x - 1) / threads.x,
                   (video[0].height + threads.y - 1) / threads.y, 1};

    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < video_length; i++)
    {
        /* H2D */
        cudaMemcpy(d_input, video[i].input.data, video[i].data_bytes, cudaMemcpyHostToDevice);
        /* KERNEL */
        GF_GLOBAL<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, video[i].width, video[i].height);
        /* D2H */
        cudaMemcpy(video[i].output.data, d_output, video[i].data_bytes, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    cudaFree(d_input);
    cudaFree(d_output);
}

void IMAGESET_NONDEFAULT_M(IO_Images *video, int video_length)
{
    uchar4 *d_input[IMAGESET_LENGTH];
    uchar4 *d_output[IMAGESET_LENGTH];
    uchar4 *h_input[IMAGESET_LENGTH];
    uchar4 *h_output[IMAGESET_LENGTH];
    /* Pinned mem (host) */
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        cudaMallocHost(&h_input[i], video[i].data_bytes);
        cudaMallocHost(&h_output[i], video[i].data_bytes);
    }
    for (int i = 0; i < video_length; i++)
        cudaMemcpy(h_input[i], video[i].input.data, video[i].data_bytes, cudaMemcpyHostToHost);
    /* I/O mem (device) */
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        cudaMalloc(&d_input[i], video[i].data_bytes);
        cudaMalloc(&d_output[i], video[i].data_bytes);
    }
    /* Streams */
    cudaStream_t streams[NUM_STREAMS_M];
    for (int i = 0; i < NUM_STREAMS_M; i++)
        cudaStreamCreate(&streams[i]);

    dim3 threads = {16, 16, 1};
    dim3 blocks = {(video[0].width + threads.x - 1) / threads.x,
                   (video[0].height + threads.y - 1) / threads.y, 1};

    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < video_length; i++)
    {
        int stream_idx = i % NUM_STREAMS_M;
        /* H2D */
        cudaMemcpyAsync(d_input[i], h_input[i], video[i].data_bytes, cudaMemcpyHostToDevice, streams[stream_idx]);
        /* KERNEL */
        GF_GLOBAL<<<blocks, threads, 0, streams[stream_idx]>>>((uchar4 *)d_input[i], (uchar4 *)d_output[i], video[i].width, video[i].height);
        /* D2H */
        cudaMemcpyAsync(h_output[i], d_output[i], video[i].data_bytes, cudaMemcpyDeviceToHost, streams[stream_idx]);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    for (int i = 0; i < NUM_STREAMS_M; i++)
        cudaStreamSynchronize(streams[i]);
    for (int i = 0; i < NUM_STREAMS_M; i++)
        cudaStreamDestroy(streams[i]);
    for (int i = 0; i < video_length; i++)
        cudaMemcpy(video[i].output.data, h_output[i], video[i].data_bytes, cudaMemcpyHostToHost);
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(h_input[i]);
        cudaFreeHost(h_output[i]);
    }
}

void IMAGESET_EXTRA(IO_Images *video, int video_length)
{
}

int main(int argc, char *argv[])
{
    IO_Images io_images_global, io_images_shared, io_images_mwpt, io_images_cpu, io_images_cmp1, io_images_cmp2;

    read_input_image_into_RGBA_image(io_images_cpu, INPUT_IMG);
    prepare_output_image(io_images_cpu);
    read_input_image_into_RGBA_image(io_images_global, INPUT_IMG);
    prepare_output_image(io_images_global);
    read_input_image_into_RGBA_image(io_images_shared, INPUT_IMG);
    prepare_output_image(io_images_shared);
    read_input_image_into_RGBA_image(io_images_mwpt, INPUT_IMG);
    prepare_output_image(io_images_mwpt);
    read_input_image_into_RGBA_image(io_images_cmp1, INPUT_IMG);
    prepare_output_image(io_images_cmp1);
    read_input_image_into_RGBA_image(io_images_cmp2, INPUT_IMG);
    prepare_output_image(io_images_cmp2);

    /******************************************************************************/
    printf("[GF-GLOBAL]\n");
    unsigned char *d_input;
    cudaMalloc((void **)&d_input, io_images_global.data_bytes);
    cudaMemcpy(d_input, io_images_global.input.data, io_images_global.data_bytes, cudaMemcpyHostToDevice);
    unsigned char *d_output;
    cudaMalloc((void **)&d_output, io_images_global.data_bytes);
    cudaMemcpyToSymbol(filter_weight, &(filter_5[FILTER_NAME][0]), 5 * 5 * sizeof(float));

    unsigned int threadx = 16;
    unsigned int thready = 16;
    dim3 threads = {threadx, thready, 1};
    dim3 blocks = {(io_images_global.width + threads.x - 1) / threads.x,
                   (io_images_global.height + threads.y - 1) / threads.y, 1};

    // a dummy run for warming up the device
    GF_GLOBAL<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_global.width, io_images_global.height);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start, _freq);
#pragma unroll
    for (int i = 0; i < VIDEO_LENGTH; i += 2)
    {
        GF_GLOBAL<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_global.width, io_images_global.height);
        GF_GLOBAL<<<blocks, threads>>>((uchar4 *)d_output, (uchar4 *)d_input, io_images_global.width, io_images_global.height);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    fprintf(stdout, "*** %.6f(ms)\n", _compute_time);

    cudaMemcpy(io_images_global.output.data, d_input, io_images_global.data_bytes, cudaMemcpyDeviceToHost);

    /******************************************************************************/
    printf("\n[GF-CPU]\n");
    unsigned char *h_input = new unsigned char[io_images_cpu.data_bytes];
    memcpy(h_input, io_images_cpu.input.data, io_images_cpu.data_bytes);
    unsigned char *h_output = new unsigned char[io_images_cpu.data_bytes];
    CHECK_TIME_START(_start, _freq);
#pragma unroll
    for (int i = 0; i < VIDEO_LENGTH; i += 2)
    {
#ifdef ENABLE_GF_CPU
        GF_CPU((uchar4 *)h_input, (uchar4 *)h_output, io_images_cpu.width, io_images_cpu.height);
        GF_CPU((uchar4 *)h_output, (uchar4 *)h_input, io_images_cpu.width, io_images_cpu.height);
#endif
    }
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    printf("*** %.6f(ms)\n", _compute_time);
    memcpy(io_images_cpu.output.data, h_input, io_images_cpu.data_bytes);

    /******************************************************************************/
    printf("\n[GF-SHARED-1]\n");
    cudaMemcpy(d_input, io_images_shared.input.data, io_images_shared.data_bytes, cudaMemcpyHostToDevice);
    threadx = 16;
    thready = 16;
    threads = {threadx, thready, 1};
    blocks = {(io_images_shared.width + threads.x - 1) / threads.x,
              (io_images_shared.height + threads.y - 1) / threads.y, 1};

    // a dummy run for warming up the device
    GF_SHARED_1<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_shared.width, io_images_shared.height);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start, _freq);
#pragma unroll
    for (int i = 0; i < VIDEO_LENGTH; i += 2)
    {
        GF_SHARED_1<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_shared.width, io_images_shared.height);
        GF_SHARED_1<<<blocks, threads>>>((uchar4 *)d_output, (uchar4 *)d_input, io_images_shared.width, io_images_shared.height);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    fprintf(stdout, "*** %.6f(ms)\n", _compute_time);
    cudaMemcpy(io_images_shared.output.data, d_input, io_images_shared.data_bytes, cudaMemcpyDeviceToHost);

    /******************************************************************************/
    printf("\n[GF-SHARED-N]\n");

    cudaMemcpy(d_input, io_images_mwpt.input.data, io_images_mwpt.data_bytes, cudaMemcpyHostToDevice);
    threads = {16 / 2, 16 / 2, 1};
    blocks = {(io_images_mwpt.width + 16 - 1) / 16,
              (io_images_mwpt.height + 16 - 1) / 16,
              1};

    // a dummy run for warming up the device
    GF_SHARED_N<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_mwpt.width, io_images_mwpt.height);
    cudaDeviceSynchronize();

    CHECK_TIME_START(_start, _freq);
#pragma unroll
    for (int i = 0; i < VIDEO_LENGTH; i += 2)
    {
        GF_SHARED_N<<<blocks, threads>>>((uchar4 *)d_input, (uchar4 *)d_output, io_images_mwpt.width, io_images_mwpt.height);
        GF_SHARED_N<<<blocks, threads>>>((uchar4 *)d_output, (uchar4 *)d_input, io_images_mwpt.width, io_images_mwpt.height);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    fprintf(stdout, "*** %.6f(ms)\n", _compute_time);
    cudaMemcpy(io_images_mwpt.output.data, d_input, io_images_mwpt.data_bytes, cudaMemcpyDeviceToHost);

    /******************************************************************************/
    threadx = 16;
    thready = 16;
    threads = {threadx, thready, 1};
    blocks = {(io_images_shared.width + threads.x - 1) / threads.x,
              (io_images_shared.height + threads.y - 1) / threads.y, 1};

    unsigned char *input1, *input2, *output;
    cudaMalloc((void **)&input1, io_images_cmp1.data_bytes);
    cudaMalloc((void **)&input2, io_images_cmp1.data_bytes);
    cudaMalloc((void **)&output, io_images_cmp1.data_bytes);

    cudaMemcpy(input1, io_images_global.output.data, io_images_global.data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(input2, io_images_shared.output.data, io_images_shared.data_bytes, cudaMemcpyHostToDevice);
    GF_COMPARE<<<blocks, threads>>>((uchar4 *)input1, (uchar4 *)input2, (uchar4 *)output, io_images_cmp1.width, io_images_cmp1.height);
    cudaMemcpy(io_images_cmp1.output.data, output, io_images_cmp1.data_bytes, cudaMemcpyDeviceToHost);

    cudaMemcpy(input1, io_images_global.output.data, io_images_global.data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(input2, io_images_mwpt.output.data, io_images_mwpt.data_bytes, cudaMemcpyHostToDevice);
    GF_COMPARE<<<blocks, threads>>>((uchar4 *)input1, (uchar4 *)input2, (uchar4 *)output, io_images_cmp2.width, io_images_cmp2.height);
    cudaMemcpy(io_images_cmp2.output.data, output, io_images_cmp2.data_bytes, cudaMemcpyDeviceToHost);
    /******************************************************************************/
    printf("\n\n\n\n");
    IO_Images video1[IMAGESET_LENGTH];
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        read_input_image_into_RGBA_image(video1[i], INPUT_IMG);
        prepare_output_image(video1[i]);
    }
    IO_Images video2[IMAGESET_LENGTH];
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        read_input_image_into_RGBA_image(video2[i], INPUT_IMG);
        prepare_output_image(video2[i]);
    }

    printf("\n[IMAGESET-DEFAULT]\n");
    cudaDeviceSynchronize();
    IMAGESET_DEFAULT(video1, IMAGESET_LENGTH);
    cudaDeviceSynchronize();
    fprintf(stdout, "*** %.6f(ms)\n", _compute_time);

    printf("\n[IMAGESET-NONDEFAULT-M]\n");
    cudaDeviceSynchronize();
    IMAGESET_NONDEFAULT_M(video2, IMAGESET_LENGTH);
    cudaDeviceSynchronize();
    fprintf(stdout, "*** %.6f(ms)\n", _compute_time);

#ifdef WRITE_FILE
    // write_filtered_data_into_output_image(io_images_global, OUTPUT_IMG("original"));
    write_filtered_data_into_output_image(io_images_cpu, OUTPUT_IMG("1a"));
    write_filtered_data_into_output_image(io_images_shared, OUTPUT_IMG("1b"));
    write_filtered_data_into_output_image(io_images_cmp1, OUTPUT_IMG("1c"));
    write_filtered_data_into_output_image(io_images_mwpt, OUTPUT_IMG("1d"));
    write_filtered_data_into_output_image(io_images_cmp2, OUTPUT_IMG("1e"));
    for (int i = 0; i < IMAGESET_LENGTH; i++)
    {
        write_filtered_data_into_output_image(video1[i], OUTPUT_IMG("2a_" + to_string(i + 1)));
        write_filtered_data_into_output_image(video2[i], OUTPUT_IMG("2b_" + to_string(i + 1)));
    }
#endif
    return 0;

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}
