#include <iostream>
#include <Eigen/Dense>
#include <cmath>


using namespace std;

using namespace Eigen;

int main()
{


	//先把p，N弄好，暂时先不做输入输出了
	int p = 0;
	int N = 0;
	//输入P,N
	cout << "请输入向量维数p：";
	cin >> p;
	cout << "\n";
	cout << "请输入观测向量组数N：";
	cin >> N;
	cout << "向量维数p=" << p << "," << "观测向量组数N=" << N << "." << endl;
	//输入数据
	MatrixXd origin_predictor(N, p);
	origin_predictor << 1, 1.46935, 1.93765, 0.90668, -1.72568, 1.15898, 2.84707, 1.33223, -2.53562, 2.75447, 1.75682, -3.34376, -0.177932, -1.56464, 1.97797,
		1, 0.117136, -1.50035, -0.00265979, -0.287058, -0.986279, -0.175744, -0.000311557, -0.0336248, 1.25104, 0.00399061, 0.430687, -0.999993, 0.000763514, -0.917598,
		1, -2.14577, 1.66381, -0.190788, 1.57, 3.60434, -3.57015, 0.409387, -3.36886, 1.76826, -0.317434, 2.61217, -0.9636, -0.299536, 1.46489,
		1, 0.464158, -0.145059, 1.12586, 1.10911, -0.784557, -0.0673301, 0.522578, 0.514803, -0.978958, -0.163316, -0.160886, 0.267567, 1.24871, 0.23013,
		1, -0.796382, -0.214268, 0.416179, 0.656488, -0.365776, 0.170639, -0.331437, -0.522815, -0.954089, -0.0891736, -0.140664, -0.826795, 0.273216, -0.569023,
		1, 0.804359, 0.247688, 0.0563595, -0.365366, -0.353007, 0.19923, 0.0453333, -0.293886, -0.93865, 0.0139596, -0.0904971, -0.996824, -0.0205919, -0.866507,
		1, 0.734795, -0.245937, 2.88826, -0.461642, -0.460076, -0.180714, 2.12228, -0.339212, -0.939515, -0.710331, 0.113535, 7.34204, -1.33334, -0.786887,
		1, 0.218678, 0.216693, 0.353765, 0.121816, -0.95218, 0.047386, 0.0773606, 0.0266385, -0.953044, 0.0766586, 0.0263967, -0.87485, 0.0430943, -0.985161,
		1, 0.105008, 0.262526, 0.0924023, 0.340339, -0.988973, 0.0275673, 0.00970298, 0.0357383, -0.93108, 0.024258, 0.0893477, -0.991462, 0.0314481, -0.88417,
		1, -0.0789081, -0.359154, 0.1699, 0.136951, -0.993774, 0.0283402, -0.0134065, -0.0108065, -0.871008, -0.0610202, -0.0491865, -0.971134, 0.0232679, -0.981244,
		1, 0.461403, 0.105987, 0.0137016, -0.287654, -0.787108, 0.0489025, 0.00632195, -0.132724, -0.988767, 0.00145219, -0.0304875, -0.999812, -0.00394132, -0.917255,
		1, -0.751, 0.019475, 0.541051, 0.763735, -0.435998, -0.0146257, -0.406329, -0.573565, -0.999621, 0.010537, 0.0148737, -0.707264, 0.413219, -0.416709,
		1, 0.415669, 0.129012, 0.452036, -0.139574, -0.827219, 0.0536261, 0.187897, -0.0580167, -0.983356, 0.0583179, -0.0180067, -0.795663, -0.0630927, -0.980519,
		1, -0.0382318, -1.13606, 0.0210645, -0.343026, -0.998538, 0.0434335, -0.000805332, 0.0131145, 0.290625, -0.0239304, 0.389697, -0.999556, -0.00722567, -0.882333,
		1, 0.228457, -0.0179151, -0.491138, -0.0606083, -0.947808, -0.00409281, -0.112204, -0.0138464, -0.999679, 0.00879877, 0.0010858, -0.758783, 0.0297671, -0.996327;
	//cout << origin_predictor << endl;
	VectorXd response_variable(N);
	response_variable << -13.803,
		-14.0973,
		-13.911,
		-14.5597,
		-14.0432,
		-14.2473,
		-13.9898,
		-14.2143,
		-14.2574,
		-14.1459,
		-14.1751,
		-14.0811,
		-14.1938,
		-14.0309,
		-14.1927;

	//cout << response_variable << endl;



	//标准化
	VectorXd mean_vector = VectorXd::Zero(p).transpose();
	VectorXd mse_vector = VectorXd::Zero(p).transpose();
	double mean = 0;//临时平均值
	double mse = 0;//临时标准差
	double response_mean = 0;
	//算平均值
	for (int i = 1; i <= p; i++) {
		mean = 0;//

		for (int j = 1; j <= N; j++) {

			mean = mean + origin_predictor(j - 1, i - 1);

		}
		mean = mean / N;
		mean_vector(i - 1) = mean;

	}

	//算标准差
	for (int i = 1; i <= p; i++) {
		mse = 0;//
		for (int j = 1; j <= N; j++) {
			mse = mse + origin_predictor(j - 1, i - 1) * origin_predictor(j - 1, i - 1);

		}
		mse = mse / N;
		mse = mse - mean_vector(i - 1) * mean_vector(i - 1);
		mse_vector(i - 1) = sqrt(mse);

	}

	//cout << "平均值向量=" << mean_vector << endl;
	//cout << "标准差向量=" << mse_vector << endl;

	response_mean = 0;
	//算响应值平均值
	for (int i = 1; i <= N; i++) {
		response_mean = response_mean + response_variable(i - 1);
	}

	response_mean = response_mean / N;


	//更新观测值

	for (int i = 1; i <= p; i++) {

		for (int j = 1; j <= N; j++) {

			if (mse_vector(i - 1) != 0) {
				origin_predictor(j - 1, i - 1) = (origin_predictor(j - 1, i - 1) - mean_vector(i - 1)) / mse_vector(i - 1);
			}
			else
			{
				origin_predictor(j - 1, i - 1) = 0;
			}
		}

	}

	//更新响应值
	for (int i = 1; i <= N; i++) {
		response_variable(i - 1) = response_variable(i - 1) - response_mean;
	}



	//输出更新结果
	cout << "标准后的原始数据为：" << origin_predictor << endl;
	cout << "更新后的响应值为：" << response_variable << endl;

	//标准化结束

	//先设定一些参数
	double lambda = 0;//待会更新为lambda_max
	double epsilon = 0.001;//常用值
	int K = 100;//常用值
	double alpha = 0.5;//alpha值，范围是[0,1]
	VectorXd beta_vector = VectorXd::Zero(p);//初始化beta向量
	double beta_temp = 0;//beta的临时值
	VectorXd residual_vector = response_variable;//启动时残差向量和响应向量的值是相同的
	double temp = 0;
	double wucha = 1;//循环终止的误差
	double flag = 1e-3;//循环终止的误差限
	clock_t start_time, end_time;
	//把lambda初始化为lambda_max;
	double max_dot = 0;
	double dot = 0;
	for (int i = 1; i <= p; i++) {
		dot = fabs(response_variable.dot(origin_predictor.col(i - 1)));
		if (dot > max_dot) { max_dot = dot; }
	}

	lambda = max_dot / (N * alpha);
	cout << "lambda_max=" << lambda << endl;
	cout << pow(2, 3.3) << endl;


	double S_function(double x, double y);
	//外边最大的循环：更新lambda
	start_time = clock();
	for (int k = 1; k <= K; k++) {
		lambda = lambda * pow(epsilon, (double)k / K);//
		wucha = 1;//先定一个肯定能使循环进行下去的参数
		while (wucha >= flag) {
			wucha = 0;
			for (int i = 1; i <= p; i++) {
				beta_temp = beta_vector(i - 1);//
				temp = (residual_vector.dot(origin_predictor.col(i - 1)));
				temp = temp / N + beta_temp;//
				temp = S_function(temp, alpha * lambda) / (1 + lambda * (1 - alpha));//
				if (temp != beta_temp)
				{
					wucha = wucha + (temp - beta_temp) * (temp - beta_temp);
					for (int j = 1; j <= N; j++) {
						//cout << j << endl;
						residual_vector(j - 1) = residual_vector(j - 1) - origin_predictor(j - 1, i - 1) * (temp - beta_temp);//更新残差（这里出过大bug）
					}
					beta_vector(i - 1) = temp;//

				}

			}
			wucha = wucha / p;

		}
		//cout << "第" << k << "轮完成。" << endl;
	}
	end_time = clock();
	cout << "坐标下降法（朴素更新法）结束，花费" << (double)(end_time - start_time) / CLOCKS_PER_SEC << "毫秒" << endl;
	cout << "得到beta向量为：" << endl;
	cout << beta_vector << endl;

	//测试区
	double preditor = 0;
	double residual = 0;
	for (int i = 1; i <= N; i++) {

		preditor = beta_vector.dot(origin_predictor.row(i - 1));
		residual = fabs(preditor - response_variable(i - 1));
		preditor = preditor + response_mean;
		cout << "第" << i << "组值为：" << response_variable(i - 1) + response_mean << ",预测值为：" << preditor << ",误差为：" << residual << "。" << endl;

	}

}


//S函数第二个参数要做到非负
double S_function(double x, double y) {
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