#include <cstdio>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "measure_host_time.h"

void prepare_input_data(float *A, int n)
{
    std::default_random_engine gen((int)"CSEG483 HW1 20211584 Junyeong JANG");
    std::uniform_real_distribution<float> fran(0.0, 1.0);
    for (int k = 0; k < n; k++)
    {
        A[k] = fran(gen);
        /* if (A[k] >= 0.6)
            A[k] = 0.3;
        // ( result value debug routine ) */
    }

    return;
}

void init_arr(float *avgsum, float *varsum, float *maximum, float *minimum, float *ori, int N)
{
    for (int i = 0; i < N; i++)
        avgsum[i] = varsum[i] = maximum[i] = minimum[i] = ori[i];

    return;
}

struct square_float
{
    __host__ __device__ float operator()(const float &x) const { return x * x; }
};

void HW1_host(int n,
              float *avgsum, float *varsum, float *maxi, float *mini,
              float *average, float *variance, float *maximum, float *minimum)
{
    for (int i = 0; i < n; i++)
        varsum[i] *= varsum[i];

    float _avgsum = 0.0,
          _varsum = 0.0,
          _maximum = -1.0,
          _minimum = 2.0;

    for (int i = 0; i < n; i++)
    {
        /* AVERAGE, VARIANCE */
        _avgsum += avgsum[i];
        _varsum += varsum[i];
        /* MAXIMUM */
        _maximum = (_maximum < maxi[i]) ? maxi[i] : _maximum;
        /* MINIMUM */
        _minimum = (_minimum > mini[i]) ? mini[i] : _minimum;
    }

    *average = (_avgsum / n);
    *variance = (_varsum / n) - ((*average) * (*average));
    *maximum = _maximum;
    *minimum = _minimum;

    return;
}

__global__ void HW1_reduce1(int n,
                            float *avgsum, float *varsum, float *maxi, float *mini,
                            bool isFirst /* for avoiding double squaring in varsum */)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x,
        stride = gridDim.x * blockDim.x;
    float _avgsum = 0.0,
          _varsum = 0.0,
          _maximum = -1.0,
          _minimum = 2.0;

    for (int i = tid; i < n; i += stride)
    {
        /* AVERAGE, VARIANCE */
        _avgsum += avgsum[i];
        if (isFirst)
            varsum[i] *= varsum[i];
        _varsum += varsum[i];
        /* MAXIMUM */
        _maximum = (_maximum < maxi[i]) ? maxi[i] : _maximum;
        /* MINIMUM */
        _minimum = (_minimum > mini[i]) ? mini[i] : _minimum;
    }

    avgsum[tid] = _avgsum;
    varsum[tid] = _varsum;
    maxi[tid] = _maximum;
    mini[tid] = _minimum;

    return;
}

void HW1_thrust(int n,
                thrust::device_vector<float> t_A,
                float *average, float *variance, float *maximum, float *minimum)
{
    float _avgsum = 0.0,
          _varsum = 0.0,
          _maximum = -1.0,
          _minimum = 2.0;

    _avgsum = thrust::reduce(t_A.begin(), t_A.end(), (float)0.0, thrust::plus<float>());
    _varsum = thrust::transform_reduce(t_A.begin(), t_A.end(), square_float(), (float)0.0, thrust::plus<float>());
    _maximum = thrust::reduce(t_A.begin(), t_A.end(), (float)-1.0, thrust::maximum<float>());
    _minimum = thrust::reduce(t_A.begin(), t_A.end(), (float)2.0, thrust::minimum<float>());

    *average = (_avgsum / n);
    *variance = (_varsum / n) - ((*average) * (*average));
    *maximum = _maximum;
    *minimum = _minimum;

    return;
}

int main(int argc, char *argv[])
{
    printf("[HW1] 20211584 Junyeong JANG\n\n");

    /***** INIT Routine *****/
    int N = 1 << 25;    /* Default : 1 << 24 */
    int blocks = 288;   /* Default : 288 */
    int threads = 1024;  /* Default : 256 */
    printf("*************************************************\n");
    printf("data size : %d\ngrid dimension : %d\nthread block dimension : %d\n", N, blocks, threads);
    printf("*************************************************\n\n");

    float *original_A = new float[N];
    prepare_input_data(original_A, N);

    float *avgsum = new float[N];
    float *varsum = new float[N];
    float *maxi = new float[N];
    float *mini = new float[N];

    float *d_avgsum, *d_varsum, *d_maxi, *d_mini, *d_temp;
    cudaMalloc((void **)&d_avgsum, N * sizeof(float));
    cudaMalloc((void **)&d_varsum, N * sizeof(float));
    cudaMalloc((void **)&d_maxi, N * sizeof(float));
    cudaMalloc((void **)&d_mini, N * sizeof(float));
    cudaMalloc((void **)&d_temp, N * sizeof(float));

    thrust::device_vector<float> t_temp(N);

    /***** EXEC Routine *****/
    // dummy run for warming up the device
    HW1_reduce1<<<blocks, threads>>>(N, d_temp, d_temp, d_temp, d_temp, true);
    HW1_reduce1<<<1, threads>>>(blocks * threads, d_temp, d_temp, d_temp, d_temp, false);
    HW1_reduce1<<<1, 1>>>(threads, d_temp, d_temp, d_temp, d_temp, false);
    cudaDeviceSynchronize();
    float dummy = thrust::reduce(t_temp.begin(), t_temp.end());
    cudaDeviceSynchronize();

    float average, variance, maximum, minimum;
    /* HOST */
    init_arr(avgsum, varsum, maxi, mini, original_A, N);

    CHECK_TIME_START(_start, _freq);
    HW1_host(N, avgsum, varsum, maxi, mini, &average, &variance, &maximum, &minimum);
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    printf("*** [1] HOST TIME : %.3f(ms)\n", _compute_time);
    printf("\taverage : %f | variance : %f\n", average, variance);
    printf("\tmaximum : %f | minimum : %f\n\n", maximum, minimum);

    /* DEVICE */
    init_arr(avgsum, varsum, maxi, mini, original_A, N);
    cudaMemcpy(d_avgsum, avgsum, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_varsum, varsum, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxi, maxi, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mini, mini, N * sizeof(float), cudaMemcpyHostToDevice);

    CHECK_TIME_START(_start, _freq);
    HW1_reduce1<<<blocks, threads>>>(N, d_avgsum, d_varsum, d_maxi, d_mini, true);
    HW1_reduce1<<<1, threads>>>(blocks * threads, d_avgsum, d_varsum, d_maxi, d_mini, false);
    HW1_reduce1<<<1, 1>>>(threads, d_avgsum, d_varsum, d_maxi, d_mini, false);
    cudaDeviceSynchronize();
    cudaMemcpy(&average, d_avgsum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&variance, d_varsum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maximum, d_maxi, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&minimum, d_mini, sizeof(float), cudaMemcpyDeviceToHost);
    average /= N;
    variance = (variance / N) - (average * average);
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    printf("*** [2] DEVICE TIME : %.3f(ms)\n", _compute_time);
    printf("\taverage : %f | variance : %f\n", average, variance);
    printf("\tmaximum : %f | minimum : %f\n\n", maximum, minimum);

    /* THRUST */
    thrust::device_vector<float> t_A(original_A, original_A + N);

    CHECK_TIME_START(_start, _freq);
    HW1_thrust(N, t_A, &average, &variance, &maximum, &minimum);
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    printf("*** [3] THRUST TIME : %.3f(ms)\n", _compute_time);
    printf("\taverage : %f | variance : %f\n", average, variance);
    printf("\tmaximum : %f | minimum : %f\n\n", maximum, minimum);

    /***** FREE ROUTINE *****/
    delete[] original_A;
    delete[] avgsum;
    delete[] varsum;
    delete[] maxi;
    delete[] mini;
    cudaFree(d_avgsum);
    cudaFree(d_varsum);
    cudaFree(d_maxi);
    cudaFree(d_mini);

    return 0;
}
