#include "compute.h"

/*
Main function called from cuc.exe
*/
int main(int argc, char* argv[]) {
	ceres::Jacobian jacobian;
	ceres::Options options;

	// Read reconstruction values from file (jacobian, number of cameras, .... )
	int numJ;
	double threshold, lambda;
	const std::string current_exec_name = argv[0];
	const std::string process_file_name = argv[2];
	std::ifstream file(current_exec_name.substr(0, current_exec_name.find("cuc.exe")) + process_file_name, std::ios_base::in);

	options._algorithm = std::stod(argv[1]);
	file >> options._lambda >> options._numCams >> options._numPoints >> options._numObs >> options._camParams;
	file >> jacobian._num_rows >> jacobian._num_cols >> numJ;
	std::vector<int> rows(jacobian._num_rows + 1);
	std::vector<int> cols(numJ);
	std::vector<double> values(numJ);

	for (int i = 0; i <= jacobian._num_rows; ++i)
		file >> rows[i];
	for (int i = 0; i < numJ; ++i)
		file >> cols[i];
	for (int i = 0; i < numJ; ++i)
		file >> values[i];

	jacobian._rows = rows;
	jacobian._cols = cols;
	jacobian._values = values;

	double* ptsUnc = (double*)malloc(6 * options._numPoints * SD);
	double* camUnc = (double*)malloc(options._camParams * options._camParams * options._numCams * SD);
	computeCovariances(&options, &jacobian,camUnc, ptsUnc);
	std::cout << "\nMain function... [done]\n";
}

/*
The function compute covariances for points and cameras.
*/
extern "C" __declspec(dllexport) void getCovariances(ceres::Options *options, ceres::Jacobian *jacobian, double* h_camUnc, double* h_ptUnc) {
	computeCovariances(options, jacobian, h_camUnc, h_ptUnc);
}