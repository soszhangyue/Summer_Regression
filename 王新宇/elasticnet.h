#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>



using namespace std;
using namespace Eigen;

#define OUTPUT_PATH_X "D:\\school\\Prime\\data\\dataX_120_2.txt"
#define OUTPUT_PATH_Y "D:\\school\\Prime\\data\\dataY_120_2.txt"

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


//��ȡ������
MatrixXd readVariable_m_x();

//��ȡ��������
MatrixXd readVariable_m_y();

//�õ���׼�����������
RegressionVar normlization(MatrixXd X, VectorXd Y);
    /*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */

    //��X�ı�׼��
MatrixXd normPredictor(MatrixXd X, NormVar normvar);

//��Y�Ĺ�һ��
VectorXd normresponse(VectorXd Y, NormVar normvar);

//������̵����ú���
double S(double z, double gamma);

//Э�������
VectorXd coordinateDescentCovariance(MatrixXd X, VectorXd Y, double alpha, double lambda);

//���ظ���
VectorXd coordinateDescentNaive(MatrixXd X, VectorXd Y, double alpha, double lambda);

//pathwiseLearning
VectorXd pathwiseLearning(MatrixXd X, VectorXd Y, double alpha);

//��Y�����һ��
VectorXd invNormResponse(VectorXd Y, NormVar normvar);
