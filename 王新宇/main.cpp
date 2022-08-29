#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>
#include "elasticnet.h"


using namespace std;
using namespace Eigen;



int main()
{
    int N, P;
    MatrixXd X;
    VectorXd Y;

    X = readVariable_m_x();
    Y = readVariable_m_y();
    N = X.rows();
    P = X.cols();
    VectorXd beta_vector = Eigen::VectorXd::Zero(P);

    cout << "X是" << endl << X << endl;
    cout << "Y是" << endl << Y << endl;

    RegressionVar a;
    a = normlization(X, Y);
    
    MatrixXd normX = normPredictor(X, a.normVar);
    VectorXd normY = normresponse(Y, a.normVar);

    cout << endl << "normX是" << endl << normX << endl;
    cout << endl << "normY是" << endl << normY << endl;

    cout << endl << "请选择您想要进行的算法：(a,b,c分别代表朴素，协方差以及pathwise)" << endl;
    string method;
    cin >> method;
    if (method == "a")
    {
        cout << endl << "请输入需要的lambda，alpha:(请用回车键隔开)" << endl;
        double lambda_input, alpha_input;
        cin >> lambda_input;
        cin >> alpha_input;
        beta_vector = coordinateDescentNaive(normX, normY, alpha_input, lambda_input);
    }
    else if (method == "b")
    {
        cout << endl << "请输入需要的lambda，alpha:(请用回车键隔开)" << endl;
        double lambda_input, alpha_input;
        cin >> lambda_input;
        cin >> alpha_input;
        beta_vector = coordinateDescentCovariance(normX, normY, alpha_input, lambda_input);
    }
    else if (method == "c")
    {
        cout << endl << "请输入需要的alpha:" << endl;
        double alpha_input;
        cin >> alpha_input;
        beta_vector = pathwiseLearning(normX, normY, alpha_input);
    }
    else
        cout << endl << "请检查输入是否正确！" << endl;


    cout << endl << "beta_vector是：" << endl << beta_vector<<endl;


    cout << endl << "原Y是：" << endl << Y << endl;


    VectorXd computY;
    computY = normX * beta_vector;
    VectorXd invnormY;
    invnormY = invNormResponse(computY, a.normVar);
    cout << endl << "逆标准的Y预测是：" << endl << invnormY<<endl;

    
    VectorXd err_Y_vector;
    err_Y_vector = invnormY - Y;
    cout << endl << "预测误差是" << endl << err_Y_vector << endl;

    double total_err_Y = 0;
    for (int i = 1; i <= N; i++)
        total_err_Y += err_Y_vector(i - 1) * err_Y_vector(i - 1);
    total_err_Y /= (double)N;
    cout << endl << "误差总量二范数计算为" << endl << total_err_Y << endl;

    return 0;
}