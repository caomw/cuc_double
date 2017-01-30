#pragma once
#include "cuc.h"

#include <cuda.h>

#include "thrust/sort.h" 
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_internal.h"
#include "magma_operators.h"
#include "testings.h"

#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Eigen>
#include <ctime>
 

#define SD sizeof(double)
#define SI sizeof(int)
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

void computeCovariances(ceres::Options *options, ceres::Jacobian *jacobian, double *camUnc, double *ptUnc);
