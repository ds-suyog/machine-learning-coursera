function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



h_LiRegre = X * theta;
%h = (1 ./ (1 .+ e.^(-h_LiRegre)));
h = sigmoid(h_LiRegre);

%theta = theta(2:length(theta))		%omitting theta0, we dont regularize theta0. 
% theta(2:length(theta))

%ones(size(y)-y) = 1-y
%y.*log(h) -> multiply each element of y (mx1) to log(h) (mx1)
J = -1/m * sum([ y.*log(h) + (1-y).*log(1-h)]) + lambda/(2*m) * sum(theta(2:length(theta)).^2);

%theta(2:length(theta)) = 1/m * lambda * theta(2:length(theta));  theta (1) = 0;   grad = 1/m * (X'* (h - y)) + theta;
%or,
theta(1)=0;
grad = 1/m * (X'* (h - y)) + 1/m * lambda * theta;



% =============================================================

end
