#include "compute.h"

using namespace std;
using namespace Eigen;

typedef SparseMatrix<double, RowMajor> SM;
typedef MatrixXd DM;

double timeDuration(clock_t from, clock_t to) {
	return double(to - from) / CLOCKS_PER_SEC;
}

void Jacobian2J(ceres::Jacobian *jacobian, SM *J) {
	J->reserve(jacobian->_cols.size());
	for (int i = 0; i < jacobian->_num_rows; i++) {
		for (int j = jacobian->_rows[i]; j < jacobian->_rows[i + 1]; ++j) {
			J->insert(i, jacobian->_cols[j]) = jacobian->_values[j];
		}
	}
}

void blockDiagonalInverseV(int numPoints, SM *V, SM *iV) {
	for (int i = 0; i < numPoints; ++i) {
		int start = i * 3;
		DM Vblock(V->block(start, start, 3, 3));
		DM iVblock(Vblock.inverse());
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				iV->coeffRef(start + j, start + k) = iVblock(j, k);
			}
		}
	}
}

void composeSchur(SM U, SM iV, SM W, SM *sZ, DM *dZ, SM *Y) {
	// schur Z from U,iV,W
	*Y = W * iV;
	SM H2((*Y) * W.transpose());
	*sZ = U - H2;
	*dZ = DM(*sZ);
}

void iUVWA(int N, DM *A, DM *iUVW) {
	double alpha = 1, beta = 0;
	DM iUVW_copy(*iUVW);
	blasf77_dsymm(lapack_side_const(MagmaLeft), lapack_uplo_const(MagmaLower), &N, &N,
		&alpha, iUVW_copy.data(), &N,
		A->data(), &N,
		&beta, iUVW->data(), &N);
}

void iZZupdate(int N, DM *iZZ) {
	double alpha = 1, beta = 0;
	blasf77_dsyrk(lapack_uplo_const(MagmaLower), lapack_trans_const(MagmaNoTrans), &N, &N,
		&alpha, iZZ->data(), &N,
		&beta, iZZ->data(), &N);
#pragma omp parallel for
	for (int j = 0; j < N; ++j) {
		for (int k = j; k < N; ++k)
			(*iZZ)(j, k) = (*iZZ)(k, j);
	}
}

void schurInverse(magma_int_t *info, int N, ceres::Options *options, DM *dZ, DM *iUVW) {
	double alpha = 1, beta = 0, change = DBL_MAX, old_change = DBL_MAX;
	// Z -> ZZ
	DM iZZ(N,N);
	blasf77_dsyrk(lapack_uplo_const(MagmaLower), lapack_trans_const(MagmaNoTrans), &N, &N,
		&alpha, dZ->data(), &N,
		&beta, iZZ.data(), &N);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j)
			iZZ(i, j) = iZZ(j, i);
	}

	// ZZ -> ZZ + lam I
	double lambda = options->_lambda;
	if (lambda == -1)
		lambda = iZZ.trace() / 1e16;		
	double s = 1 / iZZ.mean();						cout << " (our lambda: " << (s * lambda) << ") ";  //exit(1);
	iZZ = s * iZZ + s * lambda * MatrixXd::Identity(N, N);

	// ZZ -> iZZ 
	magma_dpotrf(MagmaLower, N, iZZ.data(), N, info); TESTING_CHECK(*info);
	magma_dpotri(MagmaLower, N, iZZ.data(), N, info); TESTING_CHECK(*info);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			iZZ(j, i) *= s;
			iZZ(i, j) = iZZ(j, i);
		}
	}

	// iZZ,Z -> A
	DM A(N, N);
	blasf77_dsymm(lapack_side_const(MagmaLeft), lapack_uplo_const(MagmaLower), &N, &N,
		&alpha, iZZ.data(), &N,
		dZ->data(), &N,
		&beta, A.data(), &N);


	// Taylor expansion of the x(lambda) evaluated in point x(0)
	*iUVW = MatrixXd::Identity(N, N);
	for (int i = 1; i < 10; ++i) {
		old_change = change;
		(*iUVW) += (i % 2 == 0 ? -1 : 1) * pow(lambda, i) * iZZ;
		change = abs(pow(lambda, i) * (iZZ.maxCoeff()));
		//cout << ">>> cykle " << i << ", l_inf_norm(change): " << change << "\n";
		
		if ((change < 1e-5) || (old_change < change)) { break; }
		iZZupdate(N, &iZZ);
	}
	// iUVW *= A
	iUVWA(N, &A, iUVW);
}

void schurInverseSVD(magma_int_t *info, int N, int method, DM *Z, DM *iUVW) {
	VectorXd sv(N);
	DM U(N, N), Vt(N,N);

	int lwork;
	double *hwork, *diag, *offdiag, *tauq, *taup;
	int* iwork;

	// Intel MKL Lapack SVD   ( aprox. 3-4x faster then GPU variant working with double )
	switch (method) {
	case SVD_QR_ITERATION:				// additional memory requirements: N*N + 3*N + 2*N*32 DOUBLE
		lwork = N*N + 3 * N + 2 * N * 32;
		hwork = (double*)malloc(lwork * SD);
		lapackf77_dgesvd(lapack_vec_const(MagmaAllVec), lapack_vec_const(MagmaAllVec), &N, &N,
			Z->data(), &N, sv.data(), U.data(), &N, Vt.data(), &N, hwork, &lwork, info);
		break;
	
	case SVD_DEVIDE_AND_CONQUER:		// additional memory requirements: 4*N*N + 7*N DOUBLE  + 8*N INT
		lwork = 4 * N*N + 7 * N;
		hwork = (double*)malloc(lwork * SD);
		iwork = (int*)malloc(8 * N*SI);
		lapackf77_dgesdd(lapack_vec_const(MagmaAllVec), &N, &N,
			Z->data(), &N, sv.data(), U.data(), &N, Vt.data(), &N, hwork, &lwork, iwork, info);
		break;
	}
	TESTING_CHECK(*info);


	// Combine all matrices back to the pseudo-inverse Z -> iUVW
	// U = U * diag(1/sv);   for values sv(j) > 1e-10
#pragma omp parallel for 
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)
			U(i, j) *= (sv(j) > 1e-10 ? 1/sv(j) : 0);
	}
	// iUVW = U * Vt
	double alpha = 1, beta = 0;
	blasf77_dgemm(lapack_trans_const(MagmaNoTrans), lapack_trans_const(MagmaNoTrans), &N, &N, &N,
		&alpha, U.data(), &N,
		Vt.data(), &N,
		&beta, iUVW->data(), &N);
}

void findICP(int numObs, int camParams, int numCams, int *h_Jcols, int **camsIds, int **ptsIds) {
	(*camsIds) = (int*)malloc(numObs*SI);
	(*ptsIds) = (int*)malloc(numObs*SI);
	int step = 2 * (camParams + 3);
	int camOffset = camParams * numCams;
#pragma omp parallel for
	for (int i = 0; i < numObs; ++i) {
		(*camsIds)[i] = h_Jcols[i * step] / camParams;
		(*ptsIds)[i] = (h_Jcols[i * step + camParams] - camOffset) / 3;
	}
}

void exCSPts(int numObs, int numPoints, int *ptsIds, int *maxCams, int **csPts) {
	(*maxCams) = 0;
	(*csPts) = (int*)malloc((numPoints + 1)*SI);
	memset((void*)(*csPts), 0, (numPoints + 1)*SI);
	int pid = 0, actCams;
	for (int i = 0; i < numObs; ++i) {
		if (pid != ptsIds[i]) {
			actCams = i - (*csPts)[pid];
			(*csPts)[++pid] = i;
			if (actCams >(*maxCams))
				(*maxCams) = actCams;
		}
	}
	(*csPts)[++pid] = numObs;
}

void iUVW2list(const int numCams, const int camParams, const double* h_iUVW, double* camUnc) {
	int l = 0;
	for (int i = 0; i < numCams; ++i) {
		for (int j = 0; j < camParams; ++j) {
			for (int k = j; k < camParams; ++k) {
				camUnc[l] = h_iUVW[i*camParams*(camParams*numCams + 1) + j*camParams*numCams + k];
				l++;
			}
		}
	}
}

void computeCovariances(ceres::Options *options, ceres::Jacobian *jacobian, double *camUnc, double *ptsUnc) {
	omp_set_num_threads(8);
	Eigen::setNbThreads(8);
	magma_queue_t queue;
	magma_int_t info = magma_init();
	TESTING_CHECK(info); 
	magma_queue_create(info, &queue);
	magma_print_environment();

	SM J(jacobian->_num_rows, jacobian->_num_cols);
	Jacobian2J(jacobian, &J);


	cout << "U,iV,W separation ... ";
	clock_t s1 = clock();
	int camBlockSize = options->_numCams * options->_camParams;
	int ptsBlockSize = options->_numPoints * 3;
	SM JJ(J.transpose() * J);
	SM U(JJ.topLeftCorner(camBlockSize, camBlockSize));
	SM W(JJ.topRightCorner(camBlockSize, ptsBlockSize));
	SM V(JJ.bottomRightCorner(ptsBlockSize, ptsBlockSize));
	SM iV(V);
	cout << " " << timeDuration(s1, clock()) << "s\n";


	cout << "Inverse of V ... "; 
	clock_t svi = clock();
	blockDiagonalInverseV(options->_numPoints, &V, &iV);
	cout << " " << timeDuration(svi, clock()) << "s\n";


	cout << "Schur composition ... ";
	clock_t scom = clock();
	SM Y(W.rows(), W.cols()), sZ(camBlockSize, camBlockSize);
	DM iZZ(camBlockSize, camBlockSize), iUVW(camBlockSize, camBlockSize), dZ(camBlockSize, camBlockSize);
	composeSchur(U, iV, W, &sZ, &dZ, &Y);
	cout << " " << timeDuration(scom, clock()) << "s\n";


	cout << "Camera uncertainty (Schur matrix pseudo-inverse) ...\n";
	clock_t ssch = clock();
	switch(options->_algorithm){
		case SVD_QR_ITERATION:
			schurInverseSVD(&info, camBlockSize, SVD_QR_ITERATION, &dZ, &iUVW);
			cout << "> svd qr-iteration: " << timeDuration(ssch, clock()) << "s\n";
			break;

		case SVD_DEVIDE_AND_CONQUER:
			schurInverseSVD(&info, camBlockSize, SVD_DEVIDE_AND_CONQUER, &dZ, &iUVW);
			cout << "> svd devide-and-conquer: " << timeDuration(ssch, clock()) << "s\n";
			break;

		case TAYLOR_EXPANSION:
			schurInverse(&info, camBlockSize, options, &dZ, &iUVW);
			cout << "> taylor-expansion: " << timeDuration(ssch, clock()) << "s\n";
	}


	cout << "Points uncertainty ...";
	clock_t spts = clock();
	int *camsIds, *ptsIds, *csPts, maxCams;
	findICP(options->_numObs, options->_camParams, options->_numCams, jacobian->_cols.data(), &camsIds, &ptsIds);
	thrust::sort_by_key(ptsIds, ptsIds + options->_numObs, camsIds);
	exCSPts(options->_numObs, options->_numPoints, ptsIds, &maxCams, &csPts);

	#pragma omp parallel for
	for (int i = 0; i < options->_numPoints; ++i) {
		int Ncams = csPts[i + 1] - csPts[i];
		DM WVpt(Ncams*options->_camParams, 3);
		DM iUVWpt(Ncams*options->_camParams, Ncams*options->_camParams);
		DM Cpt(3, 3);
		// W -> Wpt; iUVW -> iUVWpt
		for (int j = 0; j < Ncams; ++j) {
			int row = camsIds[csPts[i] + j] * options->_camParams;
			WVpt.block(j*options->_camParams, 0, options->_camParams, 3) = Y.block(row, i * 3, options->_camParams, 3);
			for (int k = 0; k < Ncams; ++k)
				iUVWpt.block(j*options->_camParams, k*options->_camParams, options->_camParams, options->_camParams) = 
					iUVW.block(row, camsIds[csPts[i] + k] * options->_camParams, options->_camParams, options->_camParams);
		}
		// Dense multiplication of iUVW, iUVWpt -> list of covariance of points
		Cpt = WVpt.transpose() * iUVWpt * WVpt;
		ptsUnc[i * 6 + 0] = Cpt(0, 0);
		ptsUnc[i * 6 + 1] = (Cpt(0, 1) + Cpt(1, 0)) / 2;
		ptsUnc[i * 6 + 2] = (Cpt(0, 2) + Cpt(2, 0)) / 2;
		ptsUnc[i * 6 + 3] = Cpt(1, 1);
		ptsUnc[i * 6 + 4] = (Cpt(1, 2) + Cpt(2, 1)) / 2;
		ptsUnc[i * 6 + 5] = Cpt(2, 2);
	}
	cout << " " << timeDuration(spts, clock()) << "s\n";
	cout << "Complete in time: " << timeDuration(s1, clock()) << "s";
	
	// Z+ -> list of cameras
	iUVW2list(options->_numCams, options->_camParams, iUVW.data(), camUnc);

	magma_finalize();
}