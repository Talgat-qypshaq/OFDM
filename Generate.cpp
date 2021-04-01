#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <time.h>
#include <iostream>
#include <cuComplex.h>
#include <cuda.h>
#include "Header.cuh"

double coefficientsArray[UEs];
void coefficientsFill()
{
	for(int a = 0; a<UEs; a++)
	{
		coefficientsArray[a] = (double)1/(double)UEs;
		//printf("%.5f\n", coefficientsArray[a]);
	}
}

void printSignal(cuDoubleComplex *signal, int order)
{
	for(int a = 0; a< FFT_size; a++)
	{
		printf("signal [%d]: [%d] %.2f %.2f\n", order, a, cuCreal(signal[a]), cuCimag(signal[a]));
	}
}

void printTime(double timeSpent, int label)
{
	printf("timeSpent %d %.2f\n", label, timeSpent);
}

double getGeneratedRandom()
{
	int randomNumberInt = rand() % ((100 - (-100)) + 1) - 100;
	double randomNumber = (double)randomNumberInt / 100;
	return randomNumber;
}

cuDoubleComplex getModulatedSignal()
{
	double real = getGeneratedRandom();
	double imag = getGeneratedRandom();

	if (real > 0.5)
		real = 1;
	else
		real = -1;
	if (imag > 0.5)
		imag = 1;
	else
		imag = -1;
	static cuDoubleComplex signal;
	signal = make_cuDoubleComplex(real, imag);
	return signal;
}

cuDoubleComplex decodeQAM(cuDoubleComplex signal)
{
	double real = cuCreal(signal);
	double imag = cuCimag(signal);
	//printf("REAL %.2f, IMAG %.2f\n", real, imag);
	if (real>0)
		real = 1;
	else
		real = -1;
	if (imag>0)
		imag = 1;
	else
		imag = -1;
	signal = make_cuDoubleComplex(real, imag);
	return signal;
}

cuDoubleComplex modulateSignal(cuDoubleComplex modulatedSignal)
{
	double real = cuCreal(modulatedSignal);
	double imag = cuCimag(modulatedSignal);

	if (real > 0.5)
		real = 1;
	else
		real = -1;
	if (imag > 0.5)
		imag = 1;
	else
		imag = -1;
	static cuDoubleComplex signal;
	signal = make_cuDoubleComplex(real, imag);
	return signal;
}

cuDoubleComplex * getiFFT_Main(cuDoubleComplex *inputSignal)
{
	int mem_size = sizeof(cuDoubleComplex)*FFT_size;
	cufftHandle plan;
	cufftDoubleComplex *d_signal_in, *d_signal_out;
	cudaMalloc(&d_signal_in, mem_size);
	cudaMalloc(&d_signal_out, mem_size);
	cudaMemcpy(d_signal_in, inputSignal, mem_size, cudaMemcpyHostToDevice);

	cufftPlan1d(&plan, FFT_size, CUFFT_Z2Z, BATCH);
	cufftExecZ2Z(plan, d_signal_in, d_signal_out, CUFFT_INVERSE);
	cudaDeviceSynchronize();

	cudaMemcpy(inputSignal, d_signal_out, FFT_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(d_signal_in);
	cudaFree(d_signal_out);

	return inputSignal;
}

SignalAndSpendTime getFFT_Host(cuDoubleComplex *inputSignal)
{
	int mem_size = sizeof(cuDoubleComplex)* FFT_size;
	cufftHandle plan;
	cufftDoubleComplex *d_signal_in, *d_signal_out;
	cudaMalloc(&d_signal_in, mem_size);
	cudaMalloc(&d_signal_out, mem_size);
	cudaMemcpy(d_signal_in, inputSignal, mem_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cufftPlan1d(&plan, FFT_size, CUFFT_Z2Z, BATCH);
	cufftExecZ2Z(plan, d_signal_in, d_signal_out, CUFFT_FORWARD);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(inputSignal, d_signal_out, FFT_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(d_signal_in);
	cudaFree(d_signal_out);
	SignalAndSpendTime result = {inputSignal, elapsedTime};
	return result;
}

SignalAndSpendTime getiFFT_Host(cuDoubleComplex *inputSignal)
{
	int mem_size = sizeof(cuDoubleComplex)*FFT_size;
	cufftHandle plan;
	cufftDoubleComplex *d_signal_in, *d_signal_out;
	cudaMalloc(&d_signal_in, mem_size);
	cudaMalloc(&d_signal_out, mem_size);
	cudaMemcpy(d_signal_in, inputSignal, mem_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cufftPlan1d(&plan, FFT_size, CUFFT_Z2Z, BATCH);
	cufftExecZ2Z(plan, d_signal_in, d_signal_out, CUFFT_INVERSE);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaMemcpy(inputSignal, d_signal_out, FFT_size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(d_signal_in);
	cudaFree(d_signal_out);
	/*
	for (int m = 0; m < FFT_size; m++)
	{
		char op = inputSignal[m].y < 0 ? '-' : '+';
		printf("ifft output signal: %.2f %c %.2f\n", cuCreal(inputSignal[m]), op, fabsf(cuCimag(inputSignal[m])));
	}
	*/
	SignalAndSpendTime result = { inputSignal, elapsedTime };
	return result;
}

double PIC(cuDoubleComplex *signalArray)
{
	cuDoubleComplex *signal;
	signal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cuDoubleComplex * rxCarrierSignal;
	rxCarrierSignal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cuDoubleComplex * sumTimeSignal;
	sumTimeSignal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	SignalAndSpendTime signalAndSpendTime;
	double GPUtimeSpent = 0;

	double timeFFT_1 = 0;
	double timeDecode_1 = 0;
	double timeFFT_2 = 0;
	double timeSum = 0;
	double timeSubtract = 0;
	double timeDecode_2 = 0;

	for(int k = 0; k < FFT_size; k++)
	{
		signal[k] = signalArray[k];
	}

	signalAndSpendTime = getFFT_Host(signal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeFFT_1 = timeFFT_1+signalAndSpendTime.timeSpent;
	signalAndSpendTime = divOnSqrtFFT_wrapper(rxCarrierSignal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeFFT_1 = timeFFT_1+signalAndSpendTime.timeSpent;

	signalAndSpendTime = deviceDecodeQAM_wrapper(rxCarrierSignal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeDecode_1 = timeDecode_1+signalAndSpendTime.timeSpent;
	signalAndSpendTime = deviceModulate_wrapper(rxCarrierSignal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeDecode_1 = timeDecode_1+signalAndSpendTime.timeSpent;

	signalAndSpendTime = getiFFT_Host(rxCarrierSignal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeSubtract = timeSubtract+signalAndSpendTime.timeSpent;
	signalAndSpendTime = mulOnSqrtFFT_wrapper(rxCarrierSignal);
		rxCarrierSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeSubtract = timeSubtract+signalAndSpendTime.timeSpent;

	signalAndSpendTime = sum_wrapper(rxCarrierSignal);
		sumTimeSignal = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeSum = signalAndSpendTime.timeSpent;
	//printSignal(sumTimeSignal, 1);
	signalAndSpendTime = subtract_wrapper (signal, sumTimeSignal);
		signalArray = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeSubtract = timeSubtract+signalAndSpendTime.timeSpent;
	//printSignal(signalArray, 2);
	signalAndSpendTime = getFFT_Host(signalArray);
		signalArray = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeFFT_2 = timeFFT_2+signalAndSpendTime.timeSpent;

	signalAndSpendTime = divOnSqrtFFT_wrapper(signalArray);
		signalArray = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeFFT_2 = timeFFT_2+signalAndSpendTime.timeSpent;

	signalAndSpendTime = deviceDecodeQAM_wrapper(signalArray);
		signalArray = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
				timeDecode_2 = timeDecode_2+signalAndSpendTime.timeSpent;
	//printSignal(signalArray, 3);
	printf("FFT_1 %.2f; Decode_1 %.2f; Sum %.2f; Subtract %.2f; FFT_2 %.2f; Decode_2 %.2f;\n", timeFFT_1, timeDecode_1, timeSum, timeSubtract, timeFFT_2, timeDecode_2);

	free(signal);
	free(rxCarrierSignal);
	free(sumTimeSignal);

	return GPUtimeSpent;
}

double SIC(cuDoubleComplex *signalArray)
{
	SignalAndSpendTime signalAndSpendTime;
	cuDoubleComplex *signal;
	signal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cuDoubleComplex * rxCarrierSignal;
	rxCarrierSignal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));
	cuDoubleComplex * timeSignal;
	timeSignal = (cuDoubleComplex *)malloc(FFT_size*sizeof(cuDoubleComplex));

	double GPUtimeSpent = 0;
	double timeFFT = 0;
	double timeiFFT = 0;
	double timeDecode_1 = 0;
	double timeSubtract = 0;
	double timeDecode_2 = 0;
	//skip 0 for loop because SIC iteration is until the ith UE
	for(int b = 1; b < UEs; b++)
	{
		signalAndSpendTime = getFFT_Host(signalArray);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeFFT = timeFFT + signalAndSpendTime.timeSpent;
		signalAndSpendTime = divOnSqrtFFT_wrapper(rxCarrierSignal);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeFFT = timeFFT + signalAndSpendTime.timeSpent;
		signalAndSpendTime = deviceDecodeQAM_wrapper(rxCarrierSignal);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeDecode_1 = timeDecode_1 + signalAndSpendTime.timeSpent;
		signalAndSpendTime = deviceModulate_wrapper(rxCarrierSignal);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeDecode_1 = timeDecode_1 + signalAndSpendTime.timeSpent;
		signalAndSpendTime = getiFFT_Host(rxCarrierSignal);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeiFFT = timeSubtract+signalAndSpendTime.timeSpent;
		signalAndSpendTime = mulOnSqrtFFT_wrapper(rxCarrierSignal);
			rxCarrierSignal = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeiFFT = timeSubtract+signalAndSpendTime.timeSpent;
		signalAndSpendTime = subtract_SIC_wrapper(rxCarrierSignal, signalArray, b);
			signalArray = signalAndSpendTime.inputSignal;
			GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
					timeSubtract = timeSubtract+signalAndSpendTime.timeSpent;
	}
	signalAndSpendTime = deviceDecodeQAM_wrapper(signalArray);
		signalArray = signalAndSpendTime.inputSignal;
		GPUtimeSpent = GPUtimeSpent+signalAndSpendTime.timeSpent;
			timeDecode_2 = timeDecode_2 + signalAndSpendTime.timeSpent;

		//printSignal(signalArray, 2);
		printf("FFT %.2f; Decode_1 %.2f; iFFT %.2f Subtract %.2f; Decode_2 %.2f;\n", timeFFT, timeDecode_1, timeiFFT, timeSubtract, timeDecode_2);

	free(signal);
	free(rxCarrierSignal);
	free(timeSignal);
	return GPUtimeSpent;
}
