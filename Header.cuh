#define BATCH 1
#define FFT_size 4096
#define UEs 350
extern double coefficientsArray[UEs];

struct SignalAndSpendTime
{
	cuDoubleComplex *inputSignal;
	double timeSpent;
};

double getGeneratedRandom();
void coefficientsFill();
cuDoubleComplex getModulatedSignal();
cuDoubleComplex *getiFFT_Main(cuDoubleComplex *inputSignal);
cuDoubleComplex decodeQAM(cuDoubleComplex signal);
SignalAndSpendTime getFFT_Host(cuDoubleComplex *inputSignal);
cuDoubleComplex modulateSignal(cuDoubleComplex modulatedSignal);
SignalAndSpendTime getiFFT_Host(cuDoubleComplex *inputSignal);
void printSignal(cuDoubleComplex *signal, int order);
void printTime(double timeSpent, int label);
double PIC(cuDoubleComplex *signalArray);
double SIC(cuDoubleComplex *signalArray);
extern SignalAndSpendTime deviceDecodeQAM_wrapper(cuDoubleComplex *rxCarrierSignal);
extern SignalAndSpendTime deviceModulate_wrapper(cuDoubleComplex *rxCarrierSignal);
extern SignalAndSpendTime divOnSqrtFFT_wrapper (cuDoubleComplex *inputSignal);
extern SignalAndSpendTime mulOnSqrtFFT_wrapper (cuDoubleComplex *inputSignal);
extern SignalAndSpendTime sum_wrapper (cuDoubleComplex *rxCarrierSignal);
extern SignalAndSpendTime subtract_wrapper (cuDoubleComplex *inputSignal, cuDoubleComplex *sumTimeSignal);
extern SignalAndSpendTime subtract_SIC_wrapper (cuDoubleComplex *timeSignal, cuDoubleComplex *inputSignal, int a);
