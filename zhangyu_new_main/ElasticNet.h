#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>


using namespace std;
using namespace Eigen;

#define OUTPUT_PATH "C:\\Users\\zhang\\Desktop\\Prime_intern\\data\\15_7.txt"
#define PREDICTORPATH "C:\\Users\\zhang\\Desktop\\Prime_intern\\data\\15_7_m.txt"

typedef struct {
    VectorXd meanPredictor;//X的均值
    VectorXd stdPredictor;//X的方差
    double meanResponse;//Y的均值
} NormVar;
typedef struct {
    MatrixXd predictor;//标准后的X
    VectorXd response;//去均值处理的Y
    NormVar normVar;//均值与方差
} RegressionVar;

typedef struct {
    VectorXd betaVector;//回归后的系数
    NormVar normVar;//均值与方差
} RegressedBeta;


MatrixXd readVariable(int Choice);
/*
    Description: Read space separated numeric variables from txt file.
    Input: None
    Output: Numerical matrix
    */


RegressionVar normlization(MatrixXd X, VectorXd Y);
/*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */


MatrixXd normPredictor(MatrixXd X, NormVar normvar);
/*
    Description: Normalize predictors by pre-defined mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        NormVar normvar: mean and std.
    Output:
        MatrixXd: normalized predictors.
    */

VectorXd invNormResponse(VectorXd Y, NormVar normvar);
/*
    Description: Inverse normalize responses by pre-defined mean and standard deviation.
    Input:
        VectorXd Y: normalized responses.
        NormVar normvar: mean and std.
    Output:
        MatrixXd: inverse normalized responses.
    */
//（下面两个分开的更新不用了，但是这个英文注释是自己写的，之后会添加到最后的两个总函数里）
void coordinateDescentNaive(MatrixXd X, VectorXd Y, VectorXd betaVector, double alpha, double lambda, double error_limit = 1e-3);
/*
    Description: Naively update the regression coefficient.
    Input:
        MatrixXd X: normalized predictors.
        VectorXd Y: normalized responses.
        VectorXd betaVector: pre-defined regression coefficient.
        double alpha: balance parameter alpha.
        double lambda: parameter lambda.
        double error_limit: upper error limit.
    Output:
        betaVector: regression coefficient.
    */

void coordinateDescentCovariance(MatrixXd X, VectorXd Y, VectorXd betaVector, double alpha, double lambda, double error_limit = 1e-3);
/*
Description: Update the regression coefficient by coordinateDescentCovariance.
Input:
    MatrixXd X: normalized predictors.
    VectorXd Y: normalized responses.
    VectorXd betaVector: pre-defined regression coefficient.
    double alpha: balance parameter alpha.
    double lambda: parameter lambda.
    double error_limit: upper error limit.
Output:
    betaVector: regression coefficient.
*/



//朴素更新与协方差更新（英文注释之后写）
RegressedBeta pathwiseLearning_coordinateDescentNaive(RegressionVar regressionvar, double alpha = 0.5, double error_limit = 1e-3, double epsilon = 0.001, int K = 100);

RegressedBeta pathwiseLearning_coordinateDescentCovariance(RegressionVar regressionvar, double alpha = 0.5, double error_limit = 1e-3, double epsilon = 0.001, int K = 100);
//S函数第二个参数要做到非负
double S(double x, double y);

VectorXd Predict(MatrixXd X,  RegressedBeta regressedBeta);