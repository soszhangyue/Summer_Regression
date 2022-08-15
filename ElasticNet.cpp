#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <string>
#include <regex>


using namespace std;
using namespace Eigen;

#define OUTPUT_PATH "output.txt"

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

MatrixXd readVariable(){
    /* 
    Description: Read space separated numeric variables from txt file.
    Input: None
    Output: Numerical matrix
    */
}

RegressionVar normlization(MatrixXd X, VectorXd Y){
    /* 
    Description: Normalize predictors and responses by their mean and standard deviation.
    Input: 
        MatrixXd X: predictors.
        VectorXd Y: responses.
    Output: 
        RegressionVar: normalized predictors, responses and normvar containing mean and std.
    */

}

MatrixXd normPredictor(MatrixXd X, NormVar normvar){
    /* 
    Description: Normalize predictors by pre-defined mean and standard deviation.
    Input: 
        MatrixXd X: predictors.
        NormVar normvar: mean and std.
    Output: 
        MatrixXd: normalized predictors.
    */

}

VectorXd invNormResponse(VectorXd Y, NormVar normvar){
    /* 
    Description: Inverse normalize responses by pre-defined mean and standard deviation.
    Input: 
        VectorXd Y: normalized responses.
        NormVar normvar: mean and std.
    Output: 
        MatrixXd: inverse normalized responses.
    */

}

VectorXd coordinateDescentNaive(MatrixXd X, VectorXd Y, double alpha, double lambda){

}

VectorXd coordinateDescentCovariance(MatrixXd X, VectorXd Y, double alpha, double lambda){
    
}

MatrixXd pathwiseLearning(MatrixXd X, VectorXd Y, double alpha){

}

