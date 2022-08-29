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
MatrixXd readVariable_m_x()
{
    vector<vector<double>> matrix_temp;//��������������ʵ����ԭ��������ģʽ��
    vector<double> row_data_temp;//������
    string line_temp, temp;//
    int line, col, row, i, j;//
    smatch result;//
    ifstream outputfile(OUTPUT_PATH_X);//��ֻ���ķ�ʽ���ļ����ܸı��ļ������ݡ�
    regex pattern("-[0-9]+(.[0-9]+)?|[0-9]+(.[0-9]+)?", regex::icase);//
    string::const_iterator iterStart;//
    string::const_iterator iterEnd;//
    if (!outputfile.is_open())
    {
        cout << "δ�ɹ��򿪽���ļ�:" << OUTPUT_PATH_X << endl;
    }
    line = 0;
    while (getline(outputfile, line_temp)) {//getline��ȡһ����
        if ((line_temp[0] == '!') || (line_temp.length() == 0)) {
            continue;
        }
        else {
            iterStart = line_temp.begin();
            iterEnd = line_temp.end();
            row_data_temp.clear();
            while (regex_search(iterStart, iterEnd, result, pattern)) {
                row_data_temp.push_back(atof(string(result[0]).c_str()));
                iterStart = result[0].second;
            }
            matrix_temp.push_back(row_data_temp);
            line++;
        }
    }
    outputfile.close();   //�ر��ļ�
    row = matrix_temp.size();//��������
    col = matrix_temp[0].size();//��������

    Eigen::MatrixXd matrix(row, col);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix(i, j) = matrix_temp[i][j];
        }
    }
    return matrix;
}


//��ȡ��������
MatrixXd readVariable_m_y()
{
    vector<vector<double>> matrix_temp;//��������������ʵ����ԭ��������ģʽ��
    vector<double> row_data_temp;//������
    string line_temp, temp;//
    int line, col, row, i, j;//
    smatch result;//
    ifstream outputfile(OUTPUT_PATH_Y);//��ֻ���ķ�ʽ���ļ����ܸı��ļ������ݡ�
    regex pattern("-[0-9]+(.[0-9]+)?|[0-9]+(.[0-9]+)?", regex::icase);//
    string::const_iterator iterStart;//
    string::const_iterator iterEnd;//
    if (!outputfile.is_open())
    {
        cout << "δ�ɹ��򿪽���ļ�:" << OUTPUT_PATH_Y << endl;
    }
    line = 0;
    while (getline(outputfile, line_temp)) {//getline��ȡһ����
        if ((line_temp[0] == '!') || (line_temp.length() == 0)) {
            continue;
        }
        else {
            iterStart = line_temp.begin();
            iterEnd = line_temp.end();
            row_data_temp.clear();
            while (regex_search(iterStart, iterEnd, result, pattern)) {
                row_data_temp.push_back(atof(string(result[0]).c_str()));
                iterStart = result[0].second;
            }
            matrix_temp.push_back(row_data_temp);
            line++;
        }
    }
    outputfile.close();   //�ر��ļ�
    row = matrix_temp.size();//��������
    col = matrix_temp[0].size();//��������

    Eigen::MatrixXd matrix(row, col);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix(i, j) = matrix_temp[i][j];
        }
    }
    return matrix;
}


//�õ���׼�����������
RegressionVar normlization(MatrixXd X, VectorXd Y) {
    /*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */

    int N, P;//��������
    int i, j;//�򵥼�����

    N = X.rows();
    P = X.cols();
    double dN, dP;
    dN = (double)N;
    dP = (double)P;

    VectorXd mean_vector = VectorXd::Zero(P).transpose();
    VectorXd err_vector = VectorXd::Zero(P).transpose();
    double tempmean = 0;
    double temperr = 0;

    //��Xƽ��ֵ
    for (int i = 1; i <= P; i++)
    {
        tempmean = 0;
        for (int j = 1; j <= N; j++)
            tempmean += X(j - 1, i - 1);
        tempmean /= dN;
        mean_vector(i - 1) = tempmean;
    }

    //��X��׼��
    for (int i = 1; i <= P; i++)
    {
        temperr = 0;
        for (int j = 1; j <= N; j++)
            temperr += X(j - 1, i - 1) * X(j - 1, i - 1);
        temperr /= dN;
        temperr -= mean_vector(i - 1) * mean_vector(i - 1);
        err_vector(i - 1) = sqrt(temperr);
    }

    //��Y��ƽ��ֵ
    double Y_tempmean, Y_temperr;

    /*Y_tempmean = 0;
    for (int i = 1; i <= N; i++)
    {
        Y_tempmean += Y(i - 1);
    }
    Y_tempmean /= dN;*/
    Y_tempmean = Y.sum() / (double)Y.size();

    //��Y�ı�׼��
    Y_temperr = 0;
    for (int i = 1; i <= N; i++)
        Y_temperr += Y(i - 1) * Y(i - 1);
    Y_temperr /= dN;
    Y_temperr -= Y_tempmean * Y_tempmean;

    RegressionVar regressionvar1;
    regressionvar1.predictor = X;
    regressionvar1.response = Y;

    NormVar normvar1;
    normvar1.meanPredictor = mean_vector;
    normvar1.stdPredictor = err_vector;
    normvar1.meanResponse = Y_tempmean;
    normvar1.stdResponse = Y_temperr;

    regressionvar1.normVar = normvar1;


    return regressionvar1;
}


//��X�ı�׼��
MatrixXd normPredictor(MatrixXd X, NormVar normvar) {

    int N, P;
    int i, j;

    double dN, dP;
    N = X.rows();
    P = X.cols();
    dN = (double)N;
    dP = (double)P;

    MatrixXd normX(N, P);
    VectorXd meanpredictor(N);
    VectorXd errpredictor(N);

    meanpredictor = normvar.meanPredictor;
    errpredictor = normvar.stdPredictor;


    for (int i = 1; i <= P; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            if (errpredictor(i - 1) != 0)
                normX(j - 1, i - 1) = (X(j - 1, i - 1) - meanpredictor(i - 1)) / errpredictor(i - 1);
            else
                normX(j - 1, i - 1) = 0;
        }
    }

    return normX;
}


//��Y�Ĺ�һ��
VectorXd normresponse(VectorXd Y, NormVar normvar) {
    int N;
    int i;
    double dN;

    N = Y.size();
    dN = (double)N;
    VectorXd normY(N);
    double meanY, errY;

    meanY = normvar.meanResponse;
    errY = normvar.stdResponse;

    for (int i = 1; i <= N; i++)
    {
        /*if (errY != 0)
            normY(i - 1) = (Y(i - 1) - meanY) / errY;
        else
            normY(i - 1) = 0;*/
        normY(i - 1) = Y(i - 1) - meanY;
    }
    return normY;
}


//������̵����ú���
double S(double z, double gamma) {
    if (z > gamma)
        return z - gamma;
    else if (z < -gamma)
        return z + gamma;
    else
        return 0;
}


//Э�������
VectorXd coordinateDescentCovariance(MatrixXd X, VectorXd Y, double alpha, double lambda) {

    int N, P;
    double dN, dP;
    int i, j;

    N = X.rows();
    P = X.cols();
    dN = (double)N;
    dP = (double)P;
    RegressionVar res_regressionvar;
    res_regressionvar = normlization(X, Y);

    //NormVar normvar;//?
    //normvar = res_regressionvar.normVar;//?

    VectorXd meanpredictor;
    VectorXd stdpredictor;
    double meanresponse, stdresponse;

    meanpredictor = res_regressionvar.normVar.meanPredictor;
    stdpredictor = res_regressionvar.normVar.stdPredictor;
    meanresponse = res_regressionvar.normVar.meanResponse;
    stdresponse = res_regressionvar.normVar.stdResponse;


    MatrixXd normX = normPredictor(X, res_regressionvar.normVar);//?
    VectorXd normY = normresponse(Y, res_regressionvar.normVar);//?

    VectorXd beta_vector = VectorXd::Zero(P);
    VectorXd p_gradientvector = VectorXd::Zero(P);//p�ݶ�����
    MatrixXd dot_matrix = MatrixXd::Zero(P, P);//�ڻ�����

    //��ʼ��p�ݶ�����
    /*for (int i = 1; i <= P; i++)
        p_gradientvector(i - 1) = normY.dot(normX.col(i - 1));*/

        //���ݶ�����p���ڻ�������г�ʼ��
    for (int j = 1; j <= P; j++)
    {
        for (int i = 1; i <= N; i++)
            p_gradientvector(j - 1) += normX(i - 1, j - 1) * normY(i - 1);
    }

    for (int i = 1; i <= P; i++)
    {
        for (int j = 1; j <= P; j++)
        {
            for (int l = 1; l <= N; l++)
                dot_matrix(i - 1, j - 1) += normX(l - 1, i - 1) * normX(l - 1, j - 1);
        }
    }

    double epsilon = 1e-3;
    double imprecision;
    double betaj = 0;
    double excessiveprice = 0;//��ʱ��


    clock_t start_time, end_time;

    start_time = clock();
    do
    {
        imprecision = 0;
        for (int i = 1; i <= P; i++)
        {
            betaj = beta_vector(i - 1);
            excessiveprice = p_gradientvector(i - 1);
            for (int j = 1; j <= P; j++)
            {
                if (fabs(beta_vector(j - 1)) > 0)
                    excessiveprice -= beta_vector(j - 1) * dot_matrix(i - 1, j - 1);
            }
            excessiveprice = excessiveprice / dN + betaj;
            excessiveprice = S(excessiveprice, (alpha * lambda)) / (1 + lambda * (1 - alpha));
            if (excessiveprice != betaj)
            {
                imprecision += (excessiveprice - betaj) * (excessiveprice - betaj);
                /*if (betaj != 0)
                {
                    for (int j = 1; j <= P; j++)
                        p_gradientvector(j - 1) -= dot_matrix(j - 1, i - 1) * (excessiveprice - betaj);
                }
                else
                {
                    for (int j = 1; j <= P; j++)
                    {
                        dot_matrix(j - 1, i - 1) = dot_matrix(i - 1, j - 1) = (normX.col(i - 1)).dot(normX.col(j - 1));
                        p_gradientvector(j - 1) -= dot_matrix(j - 1, i - 1) * excessiveprice;
                    }
                }*/
                beta_vector(i - 1) = excessiveprice;
            }
        }
        imprecision /= dP;

    } while (imprecision > epsilon);
    end_time = clock();
    cout << endl << "���μ������廨�ѵ�ʱ���ǣ�" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//���ظ���
VectorXd coordinateDescentNaive(MatrixXd X, VectorXd Y, double alpha, double lambda) {

    int N, P;
    double dN, dP;
    int i, j;

    N = X.rows();
    P = X.cols();
    dN = (double)N;
    dP = (double)P;
    RegressionVar res_regressionvar;
    res_regressionvar = normlization(X, Y);

    //NormVar normvar;
    //normvar = res_regressionvar.normVar;

    VectorXd meanpredictor;
    VectorXd stdpredictor;
    double meanresponse, stdresponse;

    meanpredictor = res_regressionvar.normVar.meanPredictor;
    stdpredictor = res_regressionvar.normVar.stdPredictor;
    meanresponse = res_regressionvar.normVar.meanResponse;
    stdresponse = res_regressionvar.normVar.stdResponse;


    MatrixXd normX = normPredictor(X, res_regressionvar.normVar);
    VectorXd normY = normresponse(Y, res_regressionvar.normVar);

    double epsilon = 1e-3;
    double imprecision = 0;
    double betaj = 0;
    double excessiveprice = 0;
    VectorXd beta_vector = VectorXd::Zero(P);
    VectorXd residual_vector = normY;


    clock_t start_time, end_time;

    start_time = clock();
    do
    {
        imprecision = 0;
        for (int i = 1; i <= P; i++)
        {
            betaj = beta_vector(i - 1);
            excessiveprice = (residual_vector.dot(normX.col(i - 1)));
            excessiveprice = excessiveprice / dN + betaj;
            excessiveprice = S(excessiveprice, (alpha * lambda)) / (1 + lambda * (1 - alpha));
            if (excessiveprice != betaj)
            {
                imprecision += (excessiveprice - betaj) * (excessiveprice - betaj);
                for (int j = 1; j <= N; j++)
                    residual_vector(j - 1) -= normX(j - 1, i - 1) * (excessiveprice - betaj);
                beta_vector(i - 1) = excessiveprice;
            }
        }
        imprecision /= dP;
    } while (imprecision > epsilon);
    end_time = clock();
    cout << endl << "���μ������廨�ѵ�ʱ���ǣ�" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//pathwiseLearning
VectorXd pathwiseLearning(MatrixXd X, VectorXd Y, double alpha) {
    //�����½�
    int N, P;
    double dN, dP;
    int i, j, k;

    N = X.rows();
    P = X.cols();
    dN = (double)N;
    dP = (double)P;
    RegressionVar res_regressionvar;
    res_regressionvar = normlization(X, Y);


    VectorXd meanpredictor;
    VectorXd stdpredictor;
    double meanresponse, stdresponse;

    meanpredictor = res_regressionvar.normVar.meanPredictor;
    stdpredictor = res_regressionvar.normVar.stdPredictor;
    meanresponse = res_regressionvar.normVar.meanResponse;
    stdresponse = res_regressionvar.normVar.stdResponse;


    MatrixXd normX = normPredictor(X, res_regressionvar.normVar);
    VectorXd normY = normresponse(Y, res_regressionvar.normVar);

    double lambda = 0.5;
    double skip = 0.001;
    int K = 100;

    VectorXd beta_vector = VectorXd::Zero(P);
    VectorXd p_gradientvector = VectorXd::Zero(P);
    MatrixXd dot_matrix = MatrixXd::Zero(P, P);

    double betaj = 0;
    double excessiveprice = 0;
    double epsilon = 1e-3;
    double imprecision = 0;

    //������lambdaѭ���½�
    double max_dot = 0;
    double dot = 0;
    for (int i = 1; i <= P; i++)
    {
        dot = fabs(normY.dot(normX.col(i - 1)));
        if (dot > max_dot)
            max_dot = dot;
    }
    lambda = max_dot / (dN * alpha);



    //���ݶ�����p���ڻ�������г�ʼ��
    for (int j = 1; j <= P; j++)
    {
        for (int i = 1; i <= N; i++)
            p_gradientvector(j - 1) += normX(i - 1, j - 1) * normY(i - 1);
    }

    for (int i = 1; i <= P; i++)
    {
        for (int j = 1; j <= P; j++)
        {
            for (int l = 1; l <= N; l++)
                dot_matrix(i - 1, j - 1) += normX(l - 1, i - 1) * normX(l - 1, j - 1);
        }
    }


    clock_t start_time, end_time;

    start_time = clock();
    for (int k = 1; k <= K; k++)
    {
        lambda = lambda * pow(skip, ((double)k / K));//lambdaѭ���½�
        do
        {
            imprecision = 0;
            for (int i = 1; i <= P; i++)
            {
                betaj = beta_vector(i - 1);
                excessiveprice = p_gradientvector(i - 1);
                for (int j = 1; j <= P; j++)
                {
                    if (fabs(beta_vector(j - 1)) > 0)
                        excessiveprice -= beta_vector(j - 1) * dot_matrix(i - 1, j - 1);
                }
                excessiveprice = excessiveprice / dN + betaj;
                excessiveprice = S(excessiveprice, (alpha * lambda)) / (1 + lambda * (1 - alpha));

                if (excessiveprice != betaj)
                {
                    imprecision += (excessiveprice - betaj) * (excessiveprice - betaj);
                    /*if (betaj != 0)
                    {
                        for (int j = 1; j <= P; j++)
                            p_gradientvector(j - 1) -= dot_matrix(j - 1, i - 1) * (excessiveprice - betaj);
                    }
                    else
                    {
                        for (int j = 1; j <= P; j++)
                        {
                            dot_matrix(j - 1, i - 1) = dot_matrix(i - 1, j - 1) = (normX.col(i - 1)).dot(normX.col(j - 1));
                            p_gradientvector(j - 1) -= dot_matrix(j - 1, i - 1) * excessiveprice;
                        }
                    }*/
                    //cout << "excessive" << endl << i << "\t" << excessiveprice << endl;
                    beta_vector(i - 1) = excessiveprice;
                }

            }
            imprecision /= dP;

        } while (imprecision > epsilon);

    }
    end_time = clock();
    cout << endl << "���μ������廨�ѵ�ʱ���ǣ�" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//��Y�����һ��
VectorXd invNormResponse(VectorXd Y, NormVar normvar) {
    /*
    Description: Inverse normalize responses by pre-defined mean and standard deviation.
    Input:
        VectorXd Y: normalized responses.
        NormVar normvar: mean and std.
    Output:
        MatrixXd: inverse normalized responses.
    */
    int N;
    int i;
    double dN;

    N = Y.size();
    dN = (double)N;

    double Y_mean;
    Y_mean = normvar.meanResponse;

    for (int i = 1; i <= N; i++)
        Y(i - 1) += Y_mean;

    return Y;
}
