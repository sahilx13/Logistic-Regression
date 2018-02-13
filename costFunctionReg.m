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
h = X * theta;
h = sigmoid(h);
tpow = theta.^2;
J = (((-1)* y(1)* log(h(1))) - ((1- y(1))* log(1- h(1))));
for i=2:m
    J = J + (((-1)* y(i)* log(h(i))) - ((1- y(i))* log(1- h(i)))) + ((lambda / (2*m))* sum(tpow, 1));
end
J = J / (m);
S = X' * (h-y);
S = S/m;
for i=2:length(theta)
    S(i) = S(i) + ((lambda/ m)* theta(i));
end
grad = S;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
