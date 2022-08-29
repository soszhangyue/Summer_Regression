#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>



using namespace std;
using namespace Eigen;



typedef struct {
    VectorXd meanPredictor;
    VectorXd stdPredictor;
    double meanResponse;
    double stdResponse;
} NormVar;

typedef struct {
    MatrixXd predictor;
    VectorXd response;
    NormVar normVar;
} RegressionVar;


//读取矩阵函数
MatrixXd readVariable_m_x();

//读取向量函数
MatrixXd readVariable_m_y();

//得到标准化所需的条件
RegressionVar normlization(MatrixXd X, VectorXd Y);
    /*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */

    //对X的标准化
MatrixXd normPredictor(MatrixXd X, NormVar normvar);

//对Y的归一化
VectorXd normresponse(VectorXd Y, NormVar normvar);

//计算过程的内置函数
double S(double z, double gamma);

//协方差更新
VectorXd coordinateDescentCovariance(MatrixXd X, VectorXd Y, double alpha, double lambda);

//朴素更新
VectorXd coordinateDescentNaive(MatrixXd X, VectorXd Y, double alpha, double lambda);

//pathwiseLearning
VectorXd pathwiseLearning(MatrixXd X, VectorXd Y, double alpha);

//对Y的逆归一化
VectorXd invNormResponse(VectorXd Y, NormVar normvar);
