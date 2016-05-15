function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m               = length(y); % number of training examples
J_history       = zeros(num_iters, 1);
n               = size(X,2) - 1;
gradient        = zeros(n,1);

for iter = 1:num_iters
    disp(strcat('Iteration: ',num2str( iter ) ) );
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    h           = X*theta;
    gradient    = (X'*(h-y))./m;
    
    
    disp(strcat('Value of gradient of theta_0: ',num2str( gradient(1) ) ) );
    disp(strcat('Value of gradient of theta_1: ',num2str( gradient(2) ) ) );
    
    theta(1) = theta(1) - alpha.*gradient(1);
    theta(2) = theta(2) - alpha.*gradient(2);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    disp(strcat('Value of the cost function: ',num2str( J_history(iter) ) ) );
end

end
