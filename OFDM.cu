#include "device_launch_parameters.h"
//#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <time.h>
#include "Header.cuh"
#include "cuda.h"

__global__ void divOnSqrtFFT(cuDoubleComplex *inputSignal, double *sqrtFFT_SIZE)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	//printf("CUDA [%d]: %.5f \n", index, sqrtFFT_SIZE[0]);
	if(index<FFT_size)
	{
		inputSignal[index] = cuCdiv(inputSignal[index], make_cuDoubleComplex(sqrtFFT_SIZE[0], 0));
	}
}

__global__ void mulOnSqrtFFT(cuDoubleComplex *inputSignal, double *sqrtFFT_SIZE)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index<FFT_size)
	{
		inputSignal[index] = cuCmul(inputSignal[index], make_cuDoubleComplex(sqrtFFT_SIZE[0],0));
	}
}

__global__ void deviceDecodeQAM(cuDoubleComplex *signal)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index<FFT_size)
	{
		double real = cuCreal(signal[index]);
		double imag = cuCimag(signal[index]);
		//printf("index: %d REAL %.2f, IMAG %.2f\n", index, real, imag);
		if (real>0)
			real = 1;
		else
			real = -1;
		if (imag>0)
			imag = 1;
		else
			imag = -1;
			signal[index] = make_cuDoubleComplex(real, imag);
	}
}

__global__ void deviceModulate(cuDoubleComplex *signal)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index<FFT_size)
	{
		double real = cuCreal(signal[index]);
		double imag = cuCimag(signal[index]);

		if (real > 0.5)
			real = 1;
		else
			real = -1;
		if (imag > 0.5)
			imag = 1;
		else
			imag = -1;
		signal[index] = make_cuDoubleComplex(real, imag);
	}
}

__global__ void subtract_SIC(cuDoubleComplex *timeSignal, cuDoubleComplex *signalArray, double *coefficientsArray)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index<FFT_size)
	{
		timeSignal[index]  = cuCmul(timeSignal[index], make_cuDoubleComplex(sqrt((double)coefficientsArray[0]), 0));
		timeSignal[index]  = cuCmul(timeSignal[index], make_cuDoubleComplex((double)-1, 0));
		signalArray[index] = cuCadd(signalArray[index], timeSignal[index]);
	}
}

SignalAndSpendTime deviceDecodeQAM_wrapper (cuDoubleComplex *rxCarrierSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cuDoubleComplex *dev_rxCarrierSignal;
	cudaMalloc((void**)&dev_rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_rxCarrierSignal, rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	deviceDecodeQAM << < FFT_size, UEs >> >(dev_rxCarrierSignal);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(dev_rxCarrierSignal);
	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

SignalAndSpendTime deviceModulate_wrapper (cuDoubleComplex *rxCarrierSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	cuDoubleComplex *dev_rxCarrierSignal;
	cudaMalloc((void**)&dev_rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_rxCarrierSignal, rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	deviceModulate << < 8, FFT_size, UEs >> >(dev_rxCarrierSignal);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(dev_rxCarrierSignal);
	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

SignalAndSpendTime divOnSqrtFFT_wrapper (cuDoubleComplex *inputSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	double *sqrtFFT_SIZE;
	sqrtFFT_SIZE = (double*)malloc(sizeof(double));
	sqrtFFT_SIZE[0] = sqrt(FFT_size);
	double *dev_sqrtFFT_SIZE;
	cudaMalloc((void**)&dev_sqrtFFT_SIZE, sizeof(double));
	cudaMemcpy(dev_sqrtFFT_SIZE, sqrtFFT_SIZE, sizeof(double), cudaMemcpyHostToDevice);
	cuDoubleComplex *dev_inputSignal;
	cudaMalloc((void**)&dev_inputSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_inputSignal, inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	divOnSqrtFFT << < FFT_size, UEs >> >(dev_inputSignal, dev_sqrtFFT_SIZE);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(dev_inputSignal);
	cudaFree(dev_sqrtFFT_SIZE);
	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

SignalAndSpendTime mulOnSqrtFFT_wrapper (cuDoubleComplex *inputSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;
	double *sqrtFFT_SIZE;
	sqrtFFT_SIZE = (double*)malloc(sizeof(double));
	sqrtFFT_SIZE[0] = sqrt(FFT_size);
	double *dev_sqrtFFT_SIZE;
	cudaMalloc((void**)&dev_sqrtFFT_SIZE, sizeof(double));
	cudaMemcpy(dev_sqrtFFT_SIZE, sqrtFFT_SIZE, sizeof(double), cudaMemcpyHostToDevice);
	cuDoubleComplex *dev_inputSignal;
	cudaMalloc((void**)&dev_inputSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_inputSignal, inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	mulOnSqrtFFT << < FFT_size, UEs >> >(dev_inputSignal, dev_sqrtFFT_SIZE);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(dev_inputSignal);
	cudaFree(dev_sqrtFFT_SIZE);
	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

__global__ void subtract(cuDoubleComplex *signalArray, cuDoubleComplex *sumTimeSignal)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index<FFT_size)
	{
		signalArray[index] = cuCadd(signalArray[index], cuCmul(sumTimeSignal[index], make_cuDoubleComplex((double)-1, 0)));
	}
}

SignalAndSpendTime subtract_wrapper (cuDoubleComplex *inputSignal, cuDoubleComplex *sumTimeSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;

	cuDoubleComplex *dev_inputSignal;
	cudaMalloc((void**)&dev_inputSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_inputSignal, inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	cuDoubleComplex *dev_sumSignal;
	cudaMalloc((void**)&dev_sumSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_sumSignal, sumTimeSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	//printf("test message from subtract_wrapper\n");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	subtract << < FFT_size, UEs  >> >(dev_inputSignal, dev_sumSignal);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaFree(dev_inputSignal);
	cudaFree(dev_sumSignal);

	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

__global__ void getTimeSignal(cuDoubleComplex *dev_timeSignalArray, cuDoubleComplex *rxCarrier, double *sqrtCoefficient)
{
	int b_index = blockIdx.x;
	if(b_index<FFT_size && threadIdx.x == 0)
	{
		dev_timeSignalArray[b_index] = cuCmul(rxCarrier[b_index], make_cuDoubleComplex((double)sqrtCoefficient[0], 0));
	}
	__syncthreads();
	if(b_index<FFT_size && threadIdx.x == 0)
	{
		dev_timeSignalArray[b_index] = cuCmul(dev_timeSignalArray[b_index], make_cuDoubleComplex(UEs, 0));
	}
}

SignalAndSpendTime sum_wrapper (cuDoubleComplex *rxCarrierSignal)
{
	cudaEvent_t start, stop;
	float elapsedTime;

	double *sqrtCoefficient;
	sqrtCoefficient = (double*)malloc(sizeof(double));
	sqrtCoefficient[0] = sqrt((double)coefficientsArray[0]);

	double *dev_sqrtCoefficient;
	cudaMalloc((void**)&dev_sqrtCoefficient, sizeof(double));
	cudaMemcpy(dev_sqrtCoefficient, sqrtCoefficient, sizeof(double), cudaMemcpyHostToDevice);

	cuDoubleComplex *dev_rxCarrierSignal;
	cudaMalloc((void**)&dev_rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_rxCarrierSignal, rxCarrierSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	cuDoubleComplex *dev_timeSignalArray;
	cudaMalloc((void**)&dev_timeSignalArray, FFT_size*sizeof(cuDoubleComplex));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	getTimeSignal << < FFT_size, 1  >> >(dev_timeSignalArray, dev_rxCarrierSignal, dev_sqrtCoefficient);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_timeSignalArray, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaFree(dev_rxCarrierSignal);
	cudaFree(dev_sqrtCoefficient);
	cudaFree(dev_timeSignalArray);

	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

SignalAndSpendTime subtract_SIC_wrapper (cuDoubleComplex *timeSignal, cuDoubleComplex *inputSignal, int a)
{
	cudaEvent_t start, stop;
	float elapsedTime;

	cuDoubleComplex *dev_inputSignal;
	cudaMalloc((void**)&dev_inputSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_inputSignal, inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	cuDoubleComplex *dev_timeSignal;
	cudaMalloc((void**)&dev_timeSignal, FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(dev_timeSignal, timeSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	double *sqrtCoefficient;
	sqrtCoefficient = (double*)malloc(sizeof(double));
	sqrtCoefficient[0] = sqrt((double)coefficientsArray[UEs-a-1]);

	double *dev_sqrtCoefficient;
	cudaMalloc((void**)&dev_sqrtCoefficient, sizeof(double));
	cudaMemcpy(dev_sqrtCoefficient, sqrtCoefficient, sizeof(double), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	subtract_SIC << < FFT_size, UEs >> >(dev_timeSignal, dev_inputSignal, dev_sqrtCoefficient);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cuDoubleComplex *inputSignalResult;
	inputSignalResult = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cudaMemcpy(inputSignalResult, dev_inputSignal, FFT_size*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaFree(dev_inputSignal);
	cudaFree(dev_timeSignal);
	cudaFree(dev_sqrtCoefficient);

	SignalAndSpendTime result = { inputSignalResult, elapsedTime };
	return result;
}

int main()
{
	// I. prepare NOMA signal
	srand((unsigned)time(0));
	coefficientsFill();
	cuDoubleComplex * iFFTsignal;
	iFFTsignal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));

	cuDoubleComplex * signal;
	signal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));

	double sqrtFFT = sqrt((double)FFT_size);
	double sqrtCoeff;
	sqrtCoeff = sqrt((double)coefficientsArray[1]);
	for (int b = 0; b < FFT_size; b++)
	{
		signal[b] = getModulatedSignal();
	}
	iFFTsignal = getiFFT_Main(signal);
	for (int c = 0; c < FFT_size; c++)
	{
		signal[c] = cuCmul(iFFTsignal[c], make_cuDoubleComplex(sqrtCoeff*sqrtFFT, 0));
	}
	// II. send this signal to PIC and/or SIC functions
	double PIC_time = PIC(signal);
	printf("PIC time: %.5f ms\n", PIC_time);
	//double SIC_time = SIC(signal);
	//printf("SIC time: %.5f ms\n", SIC_time);
	return 0;
}
