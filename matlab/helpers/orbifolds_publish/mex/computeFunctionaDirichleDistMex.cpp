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

void helperFcuntionalIsoDist2x2(VectorXd &pA, const VectorXd &areas, int dim, double& val, bool& flips)
{
	int block_size = dim*dim;
	int num_blocks = pA.size() / block_size;
	Map<Matrix2d> currA(pA.data(), dim, dim);
	Matrix2d currA_inv;

	// project
	val = 0;
	flips = 0;
	for (int ii = 0; ii < num_blocks; ii++)
	{
		// get current block
		new (&currA) Map<MatrixXd>(pA.data() + ii*block_size, dim, dim);
		// check inverse
		flips = flips || (currA.determinant() < 0);
		// compute inverse
		currA_inv = currA.inverse();
		val = val + areas(ii) * (currA.squaredNorm());
		// compute Tx_grad
		currA = 2 * areas(ii) * (currA);
	}
}

void helperFcuntionalIsoDist3x3(VectorXd &pA, const VectorXd &areas, int dim, double& val, bool& flips)
{
	int block_size = dim*dim;
	int num_blocks = pA.size() / block_size;
	Map<Matrix3d> currA(pA.data(), dim, dim);
	Matrix3d currA_inv;

	// project
	val = 0;
	flips = 0;
	for (int ii = 0; ii < num_blocks; ii++)
	{
		// get current block
		new (&currA) Map<MatrixXd>(pA.data() + ii*block_size, dim, dim);
		// check inverse
		flips = flips || (currA.determinant() < 0);
		// compute inverse
		currA_inv = currA.inverse();
		val = val + areas(ii) * (currA.squaredNorm() + currA_inv.squaredNorm());
		// compute Tx_grad
		currA = 2 * areas(ii) * (currA - currA_inv.transpose()*currA_inv*currA_inv.transpose());
	}
}

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray*prhs[])
{
	// assign input
	int A_rows = mxGetM(prhs[0]); // # rows of A
	int A_cols = mxGetN(prhs[0]); // # cols of A
	int areas_rows = mxGetM(prhs[1]); // # rows of A
	int areas_cols = mxGetN(prhs[1]); // # cols of A
	double *dim;
	double val;
	bool flips;
	const Map<VectorXd> A(mxGetPr(prhs[0]), A_rows, A_cols);
	const Map<VectorXd> areas(mxGetPr(prhs[1]), areas_rows, areas_cols);
	dim = mxGetPr(prhs[2]);

	if (A_cols!=1)
		mexErrMsgIdAndTxt("MATLAB:wrong_input", "first argument must be a column vector");
	if (areas_cols != 1)
		mexErrMsgIdAndTxt("MATLAB:wrong_input", "second argument must be a column vector");

	// copy
	VectorXd pA(A_rows);
	pA = A;

	// compute
	if (*dim == 2)
		helperFcuntionalIsoDist2x2(pA, areas, *dim, val, flips);
	else if (*dim == 3)
		helperFcuntionalIsoDist3x3(pA, areas, *dim, val, flips);
	else
		mexErrMsgIdAndTxt("MATLAB:wrong_dimension", "dim must be either 2 or 3");
	
	// output
	plhs[0] = mxCreateDoubleScalar(val); // functional value
	mapDenseMatrixToMex(pA, &(plhs[1])); // return Tx_grad
	plhs[2] = mxCreateLogicalScalar(flips); // were there any flips
}