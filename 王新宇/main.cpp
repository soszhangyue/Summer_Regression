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

    cout << "X��" << endl << X << endl;
    cout << "Y��" << endl << Y << endl;

    RegressionVar a;
    a = normlization(X, Y);
    
    MatrixXd normX = normPredictor(X, a.normVar);
    VectorXd normY = normresponse(Y, a.normVar);

    cout << endl << "normX��" << endl << normX << endl;
    cout << endl << "normY��" << endl << normY << endl;

    cout << endl << "��ѡ������Ҫ���е��㷨��(a,b,c�ֱ�������أ�Э�����Լ�pathwise)" << endl;
    string method;
    cin >> method;
    if (method == "a")
    {
        cout << endl << "��������Ҫ��lambda��alpha:(���ûس�������)" << endl;
        double lambda_input, alpha_input;
        cin >> lambda_input;
        cin >> alpha_input;
        beta_vector = coordinateDescentNaive(normX, normY, alpha_input, lambda_input);
    }
    else if (method == "b")
    {
        cout << endl << "��������Ҫ��lambda��alpha:(���ûس�������)" << endl;
        double lambda_input, alpha_input;
        cin >> lambda_input;
        cin >> alpha_input;
        beta_vector = coordinateDescentCovariance(normX, normY, alpha_input, lambda_input);
    }
    else if (method == "c")
    {
        cout << endl << "��������Ҫ��alpha:" << endl;
        double alpha_input;
        cin >> alpha_input;
        beta_vector = pathwiseLearning(normX, normY, alpha_input);
    }
    else
        cout << endl << "���������Ƿ���ȷ��" << endl;


    cout << endl << "beta_vector�ǣ�" << endl << beta_vector<<endl;


    cout << endl << "ԭY�ǣ�" << endl << Y << endl;


    VectorXd computY;
    computY = normX * beta_vector;
    VectorXd invnormY;
    invnormY = invNormResponse(computY, a.normVar);
    cout << endl << "���׼��YԤ���ǣ�" << endl << invnormY<<endl;

    
    VectorXd err_Y_vector;
    err_Y_vector = invnormY - Y;
    cout << endl << "Ԥ�������" << endl << err_Y_vector << endl;

    double total_err_Y = 0;
    for (int i = 1; i <= N; i++)
        total_err_Y += err_Y_vector(i - 1) * err_Y_vector(i - 1);
    total_err_Y /= (double)N;
    cout << endl << "�����������������Ϊ" << endl << total_err_Y << endl;

    return 0;
}