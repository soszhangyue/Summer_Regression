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

MatrixXd readVariable(int Choice) {
    /*
    Description: Read space separated numeric variables from txt file.
    Input: int Choice: if choice = 1, read regressionvar; else if choice = 2, read predictor. 
    Output: Numerical matrix
    */
    string Path;
    if (Choice == 1) {
        Path = OUTPUT_PATH;
    }
    else if (Choice == 2) {
        Path = PREDICTORPATH;
    }

    else {
        cout << "��������ȷ��Choice" << endl;
    }

    vector<vector<double>> matrix_temp;//��������������ʵ����ԭ��������ģʽ��
    vector<double> row_data_temp;//������
    string line_temp, temp;//
    int line, col, row, i, j;//
    smatch result;//
    ifstream outputfile(Path);//��ֻ���ķ�ʽ���ļ����ܸı��ļ������ݡ�
    regex pattern("-[0-9]+(.[0-9]+)?|[0-9]+(.[0-9]+)?", regex::icase);//
    string::const_iterator iterStart;//
    string::const_iterator iterEnd;//
    if (!outputfile.is_open())
    {
        cout << "δ�ɹ��򿪽���ļ�:" << Path << endl;
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



RegressionVar normlization(MatrixXd X, VectorXd Y) {
    /*
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output:
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */
    int N = X.rows();
    int p = X.cols();
    VectorXd mean_vector = VectorXd::Zero(p);
    VectorXd mse_vector = VectorXd::Zero(p);
    double response_mean = 0;
    RegressionVar Z;
    Z.predictor = X;
    Z.response = Y;
    mean_vector = Z.predictor.colwise().mean();
    Z.predictor.rowwise() -= mean_vector.transpose();
    mse_vector = Z.predictor.colwise().norm();
    mse_vector = mse_vector / sqrt(N);
    response_mean = Z.response.mean();
    
    for (int i = 1; i <= p; i++) {
        if (mse_vector(i - 1) != 0) {
            Z.predictor.col(i - 1) = Z.predictor.col(i - 1) / mse_vector(i - 1);
        }
        else
        {
            Z.predictor.col(i - 1) = VectorXd::Zero(N);
        }
    }

    //������Ӧֵ

    Z.response = Z.response - response_mean * VectorXd::Ones(N);
    Z.normVar.meanPredictor = mean_vector;
    Z.normVar.stdPredictor = mse_vector;
    Z.normVar.meanResponse = response_mean;

    return Z;

}

MatrixXd normPredictor(MatrixXd X, NormVar normvar) {
    /*
    Description: Normalize predictors by pre-defined mean and standard deviation.
    Input:
        MatrixXd X: predictors.
        NormVar normvar: mean and std.
    Output:
        MatrixXd: normalized predictors.
    */

    MatrixXd normalizedPredictors = X;
    //ȥ��ֵ
    normalizedPredictors.rowwise() -= normvar.meanPredictor.transpose();
    int p = normalizedPredictors.cols();
    int N = normalizedPredictors.rows();
    //�����һ
    for (int i = 1; i <= p; i++) {
        if (normvar.stdPredictor(i - 1) != 0) {
            normalizedPredictors.col(i - 1) = normalizedPredictors.col(i - 1) / normvar.stdPredictor(i - 1);
        }
        else
        {
            normalizedPredictors.col(i - 1) = VectorXd::Zero(N);
        }
    }
    return normalizedPredictors;
}

VectorXd invNormResponse(VectorXd Y, NormVar normvar) {
    /*
    Description: Inverse normalize responses by pre-defined mean and standard deviation.
    Input:
        VectorXd Y: normalized responses.
        NormVar normvar: mean and std.
    Output:
        MatrixXd: inverse normalized responses.
    */
    int N = Y.size();
    VectorXd invNormResponse = Y;
    invNormResponse = invNormResponse + normvar.meanResponse * VectorXd::Ones(N);
    return invNormResponse;
}

VectorXd coordinateDescentNaive(MatrixXd X, VectorXd Y, VectorXd betaVector,double alpha, double lambda, double error_limit = 1e-3) {
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
    double S(double x, double y);
    int N = X.rows();
    int p = X.cols();
    double beta_temp = 0;//beta����ʱֵ
    VectorXd residualVector = Y;//����ʱ�в���������Ӧ������ֵ����ͬ��
    VectorXd BetaVector = betaVector;
    double temp = 0;
    double error = 1;//ѭ����ֹ�����
    while (error >= error_limit) {
        error = 0;
        for (int i = 1; i <= p; i++) {
            beta_temp = BetaVector(i - 1);//
            temp = (residualVector.dot(X.col(i - 1)));
            temp = temp / N + beta_temp;//
            temp = S(temp, alpha * lambda) / (1 + lambda * (1 - alpha));//
            if (temp != beta_temp)
            {
                error = error + (temp - beta_temp) * (temp - beta_temp);
                for (int j = 1; j <= N; j++) {
                    residualVector(j - 1) = residualVector(j - 1) - X(j - 1, i - 1) * (temp - beta_temp);//���²в���������bug��
                }
                BetaVector(i - 1) = temp;//
            }
        }
        error = error / p;
    }
    return BetaVector;
}

VectorXd coordinateDescentCovariance(MatrixXd X, VectorXd Y, VectorXd betaVector, double alpha, double lambda, double error_limit = 1e-3) {
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
    double S(double x, double y);
    int N = X.rows();
    int p = X.cols();
    double beta_temp = 0;//beta����ʱֵ
    double temp = 0;
    double error = 1;//ѭ����ֹ�����
    VectorXd p_gradient_vector = VectorXd::Zero(p);//Э��������е�p�ݶ�����
    MatrixXd dot_Matrix = MatrixXd::Zero(p, p);//Э��������е��ڻ�����
    //��ʼ��p�ݶ�����
    for (int i = 1; i <= p; i++) {
        p_gradient_vector(i - 1) = Y.dot(X.col(i - 1));

    }
    //��ʽ����
    while (error >= error_limit) {
        error = 0;
    for (int i = 1; i <= p; i++) {
        beta_temp = betaVector(i - 1);//
        temp = p_gradient_vector(i - 1);
        temp = temp / N + beta_temp;//
        temp = S(temp, alpha * lambda) / (1 + lambda * (1 - alpha));//
        if (temp != beta_temp)
        {
            error = error + (temp - beta_temp) * (temp - beta_temp);
            if (beta_temp != 0) {
                for (int j = 1; j <= p; j++)
                {

                    p_gradient_vector(j - 1) = p_gradient_vector(j - 1) - dot_Matrix(j - 1, i - 1) * (temp - beta_temp);


                }
            }
            else {
                for (int j = 1; j <= p; j++)
                {
                    dot_Matrix(j - 1, i - 1) = dot_Matrix(i - 1, j - 1) = (X.col(i - 1)).dot(X.col(j - 1));

                    p_gradient_vector(j - 1) = p_gradient_vector(j - 1) - dot_Matrix(j - 1, i - 1) * temp;


                }
            }

            betaVector(i - 1) = temp;//

        }

    }
    error =error / p;

}

    return betaVector;
}


RegressedBeta pathwiseLearning_coordinateDescentNaive(RegressionVar regressionvar,double alpha=0.5, double error_limit = 1e-3,double epsilon =0.001,int K=100) {
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
    MatrixXd X = regressionvar.predictor;
    VectorXd Y = regressionvar.response;
    int N = X.rows();
    int p = X.cols();
    double S(double x, double y);
    double beta_temp = 0;//beta����ʱֵ
    VectorXd residualVector = Y;//����ʱ�в���������Ӧ������ֵ����ͬ��
    double temp = 0;
    double error = 1;//ѭ����ֹ�����
    //��lambda��ʼ��Ϊlambda_max;
    double max_dot = 0;
    double dot = 0;
    double lambda = 0;
    for (int i = 1; i <= p; i++) {
        dot = fabs(Y.dot(X.col(i - 1)));
        if (dot > max_dot) { max_dot = dot; }
    }

    lambda = max_dot / (N * alpha);
    cout << "lambda_max=" << lambda << endl;
    VectorXd betaVector = VectorXd::Zero(p);//��ʼ��beta����

    for (int k = 1; k <= K; k++) {
        lambda = lambda * pow(epsilon, (double)k / K);//
        error = 1;
            while (error >= error_limit) {
                error = 0;
                for (int i = 1; i <= p; i++) {
                    beta_temp = betaVector(i - 1);//
                    temp = (residualVector.dot(X.col(i - 1)));
                    temp = temp / N + beta_temp;//
                    temp = S(temp, alpha * lambda) / (1 + lambda * (1 - alpha));//
                    if (temp != beta_temp)
                    {
                        error = error + (temp - beta_temp) * (temp - beta_temp);
                        for (int j = 1; j <= N; j++) {
                            residualVector(j - 1) = residualVector(j - 1) - X(j - 1, i - 1) * (temp - beta_temp);//���²в���������bug��
                        }
                       betaVector(i - 1) = temp;//
                    }
                }
                error = error / p;
            }
    }
    RegressedBeta regressedBeta;
    regressedBeta.betaVector = betaVector;
    regressedBeta.normVar = regressionvar.normVar;
    return regressedBeta;
}


RegressedBeta pathwiseLearning_coordinateDescentCovariance(RegressionVar regressionvar, double alpha = 0.5, double error_limit = 1e-3, double epsilon = 0.001, int K = 100) {
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
    MatrixXd X = regressionvar.predictor;
    VectorXd Y = regressionvar.response;
    int N = X.rows();
    int p = X.cols();
    double S(double x, double y);
    double beta_temp = 0;//beta����ʱֵ
    double temp = 0;
    double error = 1;//ѭ����ֹ�����

    VectorXd p_gradient_vector = VectorXd::Zero(p);//Э��������е�p�ݶ�����
    MatrixXd dot_Matrix = MatrixXd::Zero(p, p);//Э��������е��ڻ�����
    //��ʼ��p�ݶ�����
    for (int i = 1; i <= p; i++) {
        p_gradient_vector(i - 1) = Y.dot(X.col(i - 1));

    }
    //��lambda��ʼ��Ϊlambda_max;
    double max_dot = 0;
    double dot = 0;
    double lambda = 0;
    for (int i = 1; i <= p; i++) {
        dot = fabs(Y.dot(X.col(i - 1)));
        if (dot > max_dot) { max_dot = dot; }
    }

    lambda = max_dot / (N * alpha);
    cout << "lambda_max=" << lambda << endl;
    VectorXd betaVector = VectorXd::Zero(p);//��ʼ��beta����

    for (int k = 1; k <= K; k++) {
        lambda = lambda * pow(epsilon, (double)k / K);//
        error = 1;
            //��ʽ����
            while (error >= error_limit) {
                error = 0;
                for (int i = 1; i <= p; i++) {
                    beta_temp = betaVector(i - 1);//
                    temp = p_gradient_vector(i - 1);
                    temp = temp / N + beta_temp;//
                    temp = S(temp, alpha * lambda) / (1 + lambda * (1 - alpha));//
                    if (temp != beta_temp)
                    {
                        error = error + (temp - beta_temp) * (temp - beta_temp);
                        if (beta_temp != 0) {
                            for (int j = 1; j <= p; j++)
                            {

                                p_gradient_vector(j - 1) = p_gradient_vector(j - 1) - dot_Matrix(j - 1, i - 1) * (temp - beta_temp);


                            }
                        }
                        else {
                            for (int j = 1; j <= p; j++)
                            {
                                dot_Matrix(j - 1, i - 1) = dot_Matrix(i - 1, j - 1) = (X.col(i - 1)).dot(X.col(j - 1));

                                p_gradient_vector(j - 1) = p_gradient_vector(j - 1) - dot_Matrix(j - 1, i - 1) * temp;


                            }
                        }

                        betaVector(i - 1) = temp;//

                    }

                }
                error = error / p;

            }

    }

    RegressedBeta regressedBeta;
    regressedBeta.betaVector = betaVector;
    regressedBeta.normVar = regressionvar.normVar;
    return regressedBeta;
}

//S�����ڶ�������Ҫ�����Ǹ�
double S(double x, double y) {
    if (x > y) {
        return x - y;
    }

    else if (x < -y) {
        return x + y;
    }

    else {
        return 0;

    }


}


VectorXd Predict(MatrixXd X,  RegressedBeta regressedBeta) {
    MatrixXd normPredictor(MatrixXd X, NormVar normvar);
    VectorXd invNormResponse(VectorXd Y, NormVar normvar);
    MatrixXd normedPredictor = normPredictor(X, regressedBeta.normVar);
    VectorXd PreditctorVector = normedPredictor *(regressedBeta.betaVector);
    VectorXd PreditctedVector = invNormResponse(PreditctorVector, regressedBeta.normVar);
    return PreditctedVector; 
}