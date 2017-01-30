#pragma once
#include <vector>
#include <iostream>

#define SVD_QR_ITERATION 0
#define SVD_DEVIDE_AND_CONQUER 1
#define TAYLOR_EXPANSION 2

namespace ceres {
	struct Jacobian {
	public:
		Jacobian() : num_rows(0), num_cols(0) {}
		int num_rows;
		int num_cols;
		std::vector<int> cols;
		std::vector<int> rows;
		std::vector<double> values;
	};

	struct Options {
	public:
		Options() {}

		Options(int numCams, int camParams, int numPoints, int numObs) : 
			_algorithm(TAYLOR_EXPANSION), _epsilon(1e-10), _lambda(-1), _numCams{ numCams }, _camParams(camParams), _numPoints(numPoints), _numObs(numObs) {}
		
		Options(int algorithm, double eps_or_lamb, int numCams, int camParams, int numPoints, int numObs) : 
			_algorithm(algorithm), _epsilon(eps_or_lamb), _lambda(eps_or_lamb), _numCams{ numCams }, _camParams(camParams), _numPoints(numPoints), _numObs(numObs) {}

		double _epsilon, _lambda;
		int _algorithm, _numCams, _camParams, _numPoints, _numObs;
	};
}

extern "C" __declspec(dllexport) void getCovariances(ceres::Options *options, ceres::Jacobian *jacobian, double* h_camUnc, double* h_ptUnc);