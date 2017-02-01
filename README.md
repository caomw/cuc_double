# Uncertatity (double)

This code allows comutation of scene quality. The computations are done in doubles on CPU due to large range of values.
The code can be compiled as a part a .mex file or running application.


## Create the project

The project mainly depends on LAPACK and BLAS libraries which calls are packaged in Magma library. It allows easy switch to GPU computation in floats. Further, the code require CUDA and MEX libraries. Please, edit the attached CMakeLists.txt i.e. fill the correct paths for required libraries. Once the project is created you can build an application. 


## Running the application

After build the project you need to have all the .dll libraries in the path.

Then, run the application by:
- `cuc.exe <_algorithm> <file_with_Jacobian_and_settings.txt>`

The parameters as (`_algorithm`, `_lambda`, ...) are described bellow. The file with Jacobian require following structure:
```
_lambda _numCams _numPoints _numObs _camParams  
_num_rows _num_cols _values.size()  
_rows  
_cols  
_values
```



## Running the mex file

The code export:
- `void getCovariances(ceres::Options \*options, ceres::Jacobian \*jacobian, double \*h_camUnc, double \*h_ptUnc);`

which can be used in another project if the project is compiled as dynamic library. It allows usage in Ceres library. 

Example of mex part in ceres:
```
ceres::Jacobian j;
j._num_cols = jacobian.num_cols;
j._num_rows = jacobian.num_rows;
j._rows = std::vector<int>(jacobian.rows.size());
j._rows = jacobian.rows;
j._cols = std::vector<int>(jacobian.cols.size());
j._cols = jacobian.cols;
j._values = std::vector<double>(jacobian.values.size());
j._values = jacobian.values;

// create an option file
int camParams = bal_problem.camera_block_size();
double *ptUnc = (double*)malloc(6 * npts * sizeof(double));
double *camUnc = (double*)malloc(camParams*camParams*ncams * sizeof(double));
ceres::Options options(TAYLOR_EXPANSION, eps_or_lamb, ncams, camParams, npts, nobs);
getCovariances(&options, &jacobian, camUnc, ptUnc);

// rewrite results to the Matlab
int NcamArr = 0.5 * camParams * (camParams+1) * ncams;   
plhs[2] = mxCreateDoubleMatrix(6*npts + NcamArr, 1, mxREAL);
double *outC = mxGetPr(plhs[2]);
for (int i = 0; i < NcamArr; i++)
  outC[i] = camUnc[i];
for (int i = 0; i < 6*npts; i++)
  outC[NcamArr + i] = ptUnc[i];
```

Example of matlab part:
```
[ ~, ~, cam_covariances ] = bundle_adjuster(ceres_interface);
```
The output array `cam_covariances` contains upper triangles of camera and then point covariance matrices which are printed in order columns first. 



## Parameters:

The application load the values from .txt file and call the same function getCovariances.

1. `ceres::Options \*options`
 1. `_algorithm`    {0-SVD_QR_ITERATION; 1-SVD_DEVIDE_AND_CONQUER; 2-TAYLOR_EXPANSION}
 2. `_epsilon`      {parameter for svd algorithms ~1e-10}
 3. `_lambda`       {-1 for authomatic selection, otherwise approximately [1e-7,1e-10]}
 4. `_numCams`      {number of cameras in reconstruction}
 5. `_camParams`    {number of camera parameters i.e. 3xposition, 3xrotation, 1xfocal, 2xradial = 9}
 6. `_numPoints`    {number of 3D points in reconstruction}
 7. `_numObs`       {number of observations in images, i.e. 2D points}
2. `ceres::Jacobian \*jacobian`    {the same structuce as in Ceres solver, more details are [here](http://ceres-solver.org/nnls_solving.html#crsmatrix)}
 1. `_num_rows`     {number of rows of Jacobian}
 2. `_num_cols`     {number of columns of Jacobian}
 3. `_rows`         {ids of start positions of rows in columns array}
 4. `_cols`         {column ids coresponding to array with values}
 5. `_values`       {double values of Jacobian}
3. `double \*h_camUnc`    {array with upper triangles of covariance matrices of cameras parameters, values are in order columns first}
4. `double \*h_ptUnc`     {array with upper triangles of covariance matrices of points parameters, values are in order columns first}
