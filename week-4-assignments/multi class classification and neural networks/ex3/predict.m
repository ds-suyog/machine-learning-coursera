function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);   %p 5000x1

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



%X 5000x401  Theta1 25x401   Theta2  10x26  y 5000x1

%adding X0 layer, column of 1s
X = [ones(size(X,1),1) X];
a_layer_1 = X;

%a_layer_2 - 5000x25
a_layer_2 = sigmoid(X* Theta1');
%including bias 1.. 5000x26
a_layer_2 = [ones(size(a_layer_2,1),1) a_layer_2];

%a_sub_3 5000x10
a_layer_3 = sigmoid(a_layer_2 * Theta2');   

fprintf('size of a_layer_3: %d %d\n',size(a_layer_3))

%a_layer_3 = max(a_layer_3,[],2)  % picking max probability per row. resultant matrix-> 5000x1 
%[val,index] = max(matr,[],2)      % index is index of first max value.. if 1,2,5,4,5 -> index=3

[val,index] = max(a_layer_3,[],2);
% these index are digits, index 3 => digit 3
a_layer_3 = index ;

%index 10 = 10, but digit is 0. So, finding indices for 10, and replace 10 with 0. This step don't work because y has value '10' for '0'. So omitted this.
%a_layer_3(find(a_layer_3==10)) = 0;

table(a_layer_3)

p=a_layer_3;


% =========================================================================


end
