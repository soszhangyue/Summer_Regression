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
    VectorXd meanPredictor;//X�ľ�ֵ
    VectorXd stdPredictor;//X�ķ���
    double meanResponse;//Y�ľ�ֵ
} NormVar;
typedef struct {
    MatrixXd predictor;//��׼���X
    VectorXd response;//ȥ��ֵ�����Y
    NormVar normVar;//��ֵ�뷽��
} RegressionVar;

typedef struct {
    VectorXd betaVector;//�ع���ϵ��
    NormVar normVar;//��ֵ�뷽��
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
//�����������ֿ��ĸ��²����ˣ��������Ӣ��ע�����Լ�д�ģ�֮�����ӵ����������ܺ����
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



//���ظ�����Э������£�Ӣ��ע��֮��д��
RegressedBeta pathwiseLearning_coordinateDescentNaive(RegressionVar regressionvar, double alpha = 0.5, double error_limit = 1e-3, double epsilon = 0.001, int K = 100);

RegressedBeta pathwiseLearning_coordinateDescentCovariance(RegressionVar regressionvar, double alpha = 0.5, double error_limit = 1e-3, double epsilon = 0.001, int K = 100);
//S�����ڶ�������Ҫ�����Ǹ�
double S(double x, double y);

VectorXd Predict(MatrixXd X,  RegressedBeta regressedBeta);