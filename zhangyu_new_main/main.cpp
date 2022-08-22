#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>
#include "ElasticNet.h"



using namespace std;

using namespace Eigen;




int main() {
    MatrixXd readSourse = readVariable();
    int N = readSourse.rows();
    int p = readSourse.cols()-1;

    RegressionVar regressionvar = normlization(readSourse.block(0,0,N,p), readSourse.col(p));
    clock_t start_time, end_time;
    start_time = clock();
    VectorXd betaVector = pathwiseLearning_coordinateDescentNaive(regressionvar);
    end_time = clock();
    cout << "坐标下降法（协方差更新法）结束，花费" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "毫秒" << endl;
    cout << "得到beta向量为：" << endl;
    cout << betaVector << endl;
    //测试区
    double preditor = 0;
    double residual = 0;
    
    for (int i = 1; i <= N; i++) {

        preditor = betaVector.dot(regressionvar.predictor.row(i - 1));
        residual = fabs(preditor - regressionvar.response(i - 1));
        preditor = preditor + regressionvar.normVar.meanResponse;
        cout << "第" << i << "组值为：" << regressionvar.response(i - 1) + regressionvar.normVar.meanResponse << ",预测值为：" << preditor << ",误差为：" << residual << "。" << endl;

    }
}