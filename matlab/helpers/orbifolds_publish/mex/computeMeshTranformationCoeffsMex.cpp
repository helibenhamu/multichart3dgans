//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//% Code implementing the paper "Accelerated Quadratic Proxy for Geometric Optimization", SIGGRAPH 2016.
//% Disclaimer: The code is provided as-is for academic use only and without any guarantees. 
//%             Please contact the author to report any bugs.
//% Written by Shahar Kovalsky (http://www.wisdom.weizmann.ac.il/~shaharko/)
//%            Meirav Galun (http://www.wisdom.weizmann.ac.il/~/meirav/)
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "mex.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include "mexHelpers.cpp"

using namespace Eigen;

void computeMeshTranformationCoeffsFullDim(const MatrixXd& F, const MatrixXd& V, SparseMatrix<double> &T, VectorXd& areas)
{
	// init
	int n_tri = F.rows();
	int d_simplex = F.cols();
	int n_vert = V.rows();
	int dim = V.cols();
	int T_rows = n_tri*dim*dim;
	int T_cols = n_vert*dim;
	int T_nnz = n_tri*dim*dim*d_simplex;


	// prepare centering matrix
	MatrixXd B = MatrixXd::Identity(d_simplex, d_simplex);
	B = B.array() - (1.0 / d_simplex);

	// prepare output matrix
	T.resize(T_cols, T_rows);
	T.reserve(VectorXi::Constant(T_rows, d_simplex));
	areas.resize(n_tri);

	// calculate differential coefficients for each element
	MatrixXd currV(d_simplex, dim);
	MatrixXd currT(dim, d_simplex);
	int curr_row = 0;
	for (int ii = 0; ii < n_tri; ii++)
	{
		// calculate current element
		for (int jj = 0; jj < d_simplex; jj++)
		{
			currV.row(jj) = V.row(F(ii, jj));
		}
		currV = B*currV; // center
		currT = currV.fullPivLu().solve(B); // solver

		// fill into the correct places of T
		for (int cd = 0; cd < dim; cd++)
		for (int cr = 0; cr < dim; cr++)
		{
			for (int cc = 0; cc < d_simplex; cc++)
				T.insert(F(ii, cc) + (cd*n_vert), curr_row) = currT(cr, cc);
			curr_row += 1;
		}

		// calculate area
		areas(ii) = (currV.bottomRows(dim).rowwise() - currV.row(0)).determinant() / 2;
	}

	// compress
	T.makeCompressed();
	T = T.transpose();
}


void orth(const MatrixXd &A, MatrixXd &Q)
{

	//perform svd on A = U*S*V' (V is not computed and only the thin U is computed)
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU);
	Eigen::MatrixXd U = svd.matrixU();
	const Eigen::VectorXd S = svd.singularValues();

	//get rank of A
	int m = A.rows();
	int n = A.cols();
	double tol = std::max(m, n) * S.maxCoeff() *  2.2204e-16;
	int r = 0;
	for (int i = 0; i < S.rows(); ++r, ++i)
	{
		if (S[i] < tol)
			break;
	}

	//keep r first columns of U
	Q = U.block(0, 0, U.rows(), r);
}

void compute2dEmbedding(const MatrixXd& V, MatrixXd& A)
{
	// given a nXn matrix whose columns are the vertices of a (n-1)-D simplex,
	// returns the transformation A, s.t A*V gives embedding in (n-1)-D

	MatrixXd ctrV(V.rows() - 1, V.cols());
	ctrV = -V.bottomRows(V.rows() - 1);
	ctrV.rowwise() += V.row(0);
	ctrV.transpose();
	orth(ctrV, A);
	//if (((ctrV*A).determinant()) < 0)
	//	A.col(0).swap(A.col(1));
	A.transpose();
}

void embedTriangle(const MatrixXd& V, MatrixXd& flatV, double& area)
{
	VectorXd v1 = V.row(1) - V.row(0);
	VectorXd v2 = V.row(2) - V.row(0);

	double norm_v1 = v1.norm();
	double norm_v2 = v2.norm();
	double cos_theta = v1.dot(v2) / (norm_v1*norm_v2);
	double sin_theta = sqrt(1 - cos_theta*cos_theta);

	flatV << 0, 0,
		norm_v1, 0,
		norm_v2*cos_theta, norm_v2*sin_theta;

	area = norm_v1*norm_v2*sin_theta / 2;
}

void computeMeshTranformationCoeffsFlatenning(const MatrixXd& F, const MatrixXd& V, SparseMatrix<double> &T, VectorXd& areas)
{
	// init
	int n_tri = F.rows();
	int d_simplex = F.cols();
	int n_vert = V.rows();
	int dim = V.cols();
	int d_diff = dim - 1;
	int T_rows = n_tri*d_diff*d_diff;
	int T_cols = n_vert*d_diff;
	int T_nnz = n_tri*d_diff*d_diff*d_simplex;

	assert(d_simplex == 3 && dim == 3);

	// prepare centering matrix
	MatrixXd B = MatrixXd::Identity(d_simplex, d_simplex);
	B = B.array() - (1.0 / d_simplex);

	// prepare output matrix
	T.resize(T_cols, T_rows);
	T.reserve(VectorXi::Constant(T_rows, d_simplex));
	areas.resize(n_tri);

	// calculate differential coefficients for each element
	MatrixXd currV(d_simplex, dim);
	MatrixXd currT(dim, d_simplex);
	MatrixXd RFlat(dim, d_diff);
	MatrixXd currVFlat(d_simplex, d_diff);
	int curr_row = 0;
	for (int ii = 0; ii < n_tri; ii++)
	{
		// calculate current element
		for (int jj = 0; jj < d_simplex; jj++)
		{
			currV.row(jj) = V.row(F(ii, jj));
		}
		// transform to plane
		embedTriangle(currV, currVFlat, areas(ii)); // this only works for triangles
		// compute
		currVFlat = B*currVFlat; // center
		currT = currVFlat.fullPivLu().solve(B); // solver

		// fill into the correct places of T
		for (int cd = 0; cd < d_diff; cd++)
		for (int cr = 0; cr < d_diff; cr++)
		{
			for (int cc = 0; cc < d_simplex; cc++)
				T.insert(F(ii, cc) + (cd*n_vert), curr_row) = currT(cr, cc);
			curr_row += 1;
		}
	}

	// compress
	T.makeCompressed();
	T = T.transpose();
}

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray*prhs[])
{
	// assign input
	int n_tri = mxGetM(prhs[0]); // # rows of F
	int d_simplex = mxGetN(prhs[0]); // # cols of F
	int n_vert = mxGetM(prhs[1]); // # rows of V
	int dim = mxGetN(prhs[1]); // # cols of V
	const Map<MatrixXd, Aligned> Fmatlab(mxGetPr(prhs[0]), n_tri, d_simplex);
	const Map<MatrixXd, Aligned> V(mxGetPr(prhs[1]), n_vert, dim);
	
	// update index numbers to 0-base
	MatrixXd F (Fmatlab);
	F = F.array() - 1;	

	// compute
	SparseMatrix<double> T;
	VectorXd areas;
	if (d_simplex == 3 && dim == 2)
	{
		// Planar triangulation
		computeMeshTranformationCoeffsFullDim(F, V, T, areas);
	}
	else if (d_simplex == 4 && dim == 3)
	{
		// Tet mesh
		computeMeshTranformationCoeffsFullDim(F, V, T, areas);
	}
	else if (d_simplex == 3 && dim == 3)
	{
		// 3D surface
		computeMeshTranformationCoeffsFlatenning(F, V, T, areas);
	}
	else
		mexErrMsgIdAndTxt("MATLAB:invalidInputs", "Invalid input dimensions or mesh type not supported");


	// assign outputs
	mapSparseMatrixToMex(T, &(plhs[0]));
	mapDenseMatrixToMex(areas, &(plhs[1]));
}