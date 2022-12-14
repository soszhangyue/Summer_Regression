#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>


using namespace std;
using namespace Eigen;

#define OUTPUT_PATH_X "D:\\school\\Prime\\data\\dataX_120_1.txt"
#define OUTPUT_PATH_Y "D:\\school\\Prime\\data\\dataY_120_1.txt"

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
MatrixXd readVariable_m_x()
{
    vector<vector<double>> matrix_temp;//列向量（这里其实还是原来的数组模式）
    vector<double> row_data_temp;//行向量
    string line_temp, temp;//
    int line, col, row, i, j;//
    smatch result;//
    ifstream outputfile(OUTPUT_PATH_X);//以只读的方式打开文件不能改变文件的内容。
    regex pattern("-[0-9]+(.[0-9]+)?|[0-9]+(.[0-9]+)?", regex::icase);//
    string::const_iterator iterStart;//
    string::const_iterator iterEnd;//
    if (!outputfile.is_open())
    {
        cout << "未成功打开结果文件:" << OUTPUT_PATH_X << endl;
    }
    line = 0;
    while (getline(outputfile, line_temp)) {//getline读取一整行
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
    outputfile.close();   //关闭文件
    row = matrix_temp.size();//（行数）
    col = matrix_temp[0].size();//（列数）

    Eigen::MatrixXd matrix(row, col);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix(i, j) = matrix_temp[i][j];
        }
    }
    return matrix;
}


//读取向量函数
MatrixXd readVariable_m_y()
{
    vector<vector<double>> matrix_temp;//列向量（这里其实还是原来的数组模式）
    vector<double> row_data_temp;//行向量
    string line_temp, temp;//
    int line, col, row, i, j;//
    smatch result;//
    ifstream outputfile(OUTPUT_PATH_Y);//以只读的方式打开文件不能改变文件的内容。
    regex pattern("-[0-9]+(.[0-9]+)?|[0-9]+(.[0-9]+)?", regex::icase);//
    string::const_iterator iterStart;//
    string::const_iterator iterEnd;//
    if (!outputfile.is_open())
    {
        cout << "未成功打开结果文件:" << OUTPUT_PATH_Y << endl;
    }
    line = 0;
    while (getline(outputfile, line_temp)) {//getline读取一整行
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
    outputfile.close();   //关闭文件
    row = matrix_temp.size();//（行数）
    col = matrix_temp[0].size();//（列数）

    Eigen::MatrixXd matrix(row, col);

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix(i, j) = matrix_temp[i][j];
        }
    }
    return matrix;
}


//得到标准化所需的条件
RegressionVar normlization(MatrixXd X, VectorXd Y) {
    /*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */

    int N, P;//矩阵行列
    int i, j;//简单计数器

    N = X.rows();
    P = X.cols();
    double dN, dP;
    dN = (double)N;
    dP = (double)P;

    VectorXd mean_vector = VectorXd::Zero(P).transpose();
    VectorXd err_vector = VectorXd::Zero(P).transpose();
    double tempmean = 0;
    double temperr = 0;

    //算X平均值
    /*for (int i = 1; i <= P; i++)
    {
        tempmean = 0;
        for (int j = 1; j <= N; j++)
            tempmean += X(j - 1, i - 1);
        tempmean /= dN;
        mean_vector(i - 1) = tempmean;
    }*/
    for (int i = 1; i <= P; i++)
    {
        tempmean = 0;
        tempmean += X.col(i - 1).mean();
        mean_vector(i - 1) = tempmean;
    }

    //算X标准差
    /*for (int i = 1; i <= P; i++)
    {
        temperr = 0;
        for (int j = 1; j <= N; j++)
            temperr += X(j - 1, i - 1) * X(j - 1, i - 1);
        temperr /= dN;
        temperr -= mean_vector(i - 1) * mean_vector(i - 1);
        err_vector(i - 1) = sqrt(temperr);
    }*/
    for (int i = 1; i <= P; i++)
    {
        temperr = 0;
        temperr += (X.col(i - 1).norm())*(X.col(i-1).norm());
        temperr /= dN;
        temperr -= mean_vector(i - 1) * mean_vector(i - 1);
        err_vector(i - 1) = sqrt(temperr);
    }


    //算Y的平均值
    double Y_tempmean, Y_temperr;

    /*Y_tempmean = 0;
    for (int i = 1; i <= N; i++)
    {
        Y_tempmean += Y(i - 1);
    }
    Y_tempmean /= dN;*/
    Y_tempmean = Y.mean();

    //算Y的标准差
    /*Y_temperr = 0;
    for (int i = 1; i <= N; i++)
        Y_temperr += Y(i - 1) * Y(i - 1);
    Y_temperr /= dN;
    Y_temperr -= Y_tempmean * Y_tempmean;*/
    Y_temperr = Y.norm();
    Y_temperr /= dN;
    Y_temperr -= Y_tempmean * Y_tempmean;
    Y_temperr = sqrt(Y_temperr);


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


//对X的标准化
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


//对Y的归一化
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


//计算过程的内置函数
double S(double z, double gamma) {
    if (z > gamma)
        return z - gamma;
    else if (z < -gamma)
        return z + gamma;
    else
        return 0;
}


//协方差更新
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
    VectorXd p_gradientvector = VectorXd::Zero(P);//p梯度向量
    MatrixXd dot_matrix = MatrixXd::Zero(P, P);//内积矩阵


        //对梯度向量p和内积矩阵进行初始化
    /*for (int j = 1; j <= P; j++)
    {
        for (int i = 1; i <= N; i++)
            p_gradientvector(j - 1) += normX(i - 1, j - 1) * normY(i - 1);
    }*/
    for (int j = 1; j <= P; j++)
    {
        p_gradientvector(j - 1) += normY.dot(normX.col(j - 1));
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
    double excessiveprice = 0;//临时量


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
    cout << endl << "本次计算主体花费的时间是：" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//朴素更新
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
                /*for (int j = 1; j <= N; j++)
                    residual_vector(j - 1) -= normX(j - 1, i - 1) * (excessiveprice - betaj);*/
                residual_vector -= normX.col(i - 1) * (excessiveprice - betaj);
                beta_vector(i - 1) = excessiveprice;
            }
        }
        imprecision /= dP;
    } while (imprecision > epsilon);
    end_time = clock();
    cout << endl << "本次计算主体花费的时间是：" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//pathwiseLearning
VectorXd pathwiseLearning(MatrixXd X, VectorXd Y, double alpha) {
    //坐标下降
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

    //先来求lambda循环下降
    double max_dot = 0;
    double dot = 0;
    for (int i = 1; i <= P; i++)
    {
        dot = fabs(normY.dot(normX.col(i - 1)));
        if (dot > max_dot)
            max_dot = dot;
    }
    lambda = max_dot / (dN * alpha);



    //对梯度向量p和内积矩阵进行初始化
    /*for (int j = 1; j <= P; j++)
    {
        for (int i = 1; i <= N; i++)
            p_gradientvector(j - 1) += normX(i - 1, j - 1) * normY(i - 1);
    }*/
    for (int j = 1; j <= P; j++)
    {
        p_gradientvector(j - 1) += normY.dot(normX.col(j - 1));
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
        lambda = lambda * pow(skip, ((double)k / K));//lambda循环下降
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
    cout << endl << "本次计算主体花费的时间是：" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    return beta_vector;
}


//对Y的逆归一化
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
