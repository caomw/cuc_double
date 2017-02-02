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
	const std::string current_dir = current_exec_name.substr(0, current_exec_name.find("cuc.exe"));
	std::ifstream file(current_dir + process_file_name, std::ios_base::in);

	options._algorithm = std::stod(argv[1]);
	file >> options._lambda >> options._numCams >> options._camParams >> options._numPoints >> options._numObs ;
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

	// alocate output arrays
	int num_camera_covar_values = 0.5 * options._camParams * (options._camParams + 1);
	int camUnc_size = num_camera_covar_values * options._numCams;
	double* camUnc = (double*)malloc(camUnc_size * SD);
	double* ptsUnc = (double*)malloc(6 * options._numPoints * SD);

	// run the main code
	computeCovariances(&options, &jacobian,camUnc, ptsUnc);
	
	std::cout << "\nPrinting the results to file... [done]\n";
	std::ofstream outfile(current_dir + process_file_name.substr(0, current_exec_name.find(".")+1) + std::string("_covariances.txt"));
	outfile << options._lambda << " " << options._numCams << " " << options._camParams << " " << options._numPoints << " " << options._numObs << "\n";
	for (int i = 0; i <= camUnc_size; ++i)
		outfile << camUnc[i] << " ";
	outfile << "\n";
	for (int i = 0; i < (6 * options._numPoints); ++i)
		outfile << ptsUnc[i] << " ";
	outfile.close();

	std::cout << "Main function... [done]\n";
}

/*
The function compute covariances for points and cameras.
*/
extern "C" __declspec(dllexport) void getCovariances(ceres::Options *options, ceres::Jacobian *jacobian, double* h_camUnc, double* h_ptUnc) {
	computeCovariances(options, jacobian, h_camUnc, h_ptUnc);
}