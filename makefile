NVCC=nvcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f
all: OFDM

main: OFDM.o Generate.o
	${NVCC} ${OPT} -o main OFDM.o Generate.o

Generate.o: Header.cuh Generate.cpp
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c Generate.cpp

OFDM.o: Header.cuh OFDM.cu
	$(NVCC) ${OPT} $(CUDAFLAGS)	-std=c++11 -c OFDM.cu -lcufft

OFDM: OFDM.o Generate.o
	${NVCC} ${CUDAFLAGS} -o OFDM OFDM.o Generate.o -lcufft
clean:
	${RM} *.o OFDM
