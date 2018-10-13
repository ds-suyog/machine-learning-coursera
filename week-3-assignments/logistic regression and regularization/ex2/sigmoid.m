function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%alternative: for i = 1:size(z,1), j = 1:size(z,2) --> z(i,j) = 1/(1+e^(-z(i,j)))
%using anonymous function: g = arrayfun(@(x) 1/(1+e^(-x)), z)

g = (1 ./ (1 .+ e.^(-z)));


% =============================================================

end
