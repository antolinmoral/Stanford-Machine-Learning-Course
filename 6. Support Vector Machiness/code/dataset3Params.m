function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_v         = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_v     = [0.01 0.03 0.1 0.3 1 3 10 30];
num_casos   = length(C_v);

C_m         = repmat(C_v',1,num_casos);
sigma_m     = repmat(sigma_v,num_casos,1);
mse         = zeros(num_casos, num_casos);

for ii = 1:num_casos  %filas
    for jj=1:num_casos   %columnas
        model       = svmTrain(X, y, C_m(ii,jj), @(x1, x2) gaussianKernel(x1, x2, sigma_m(ii,jj)));
        pred        = svmPredict(model, Xval);
        mse(ii,jj)  = mean(double(pred ~= yval));
    end
end
minError        = min(min(mse));
indMinError     = mse == minError;
C               = C_m(indMinError); 
sigma           = sigma_m(indMinError);



% =========================================================================

end
