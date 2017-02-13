function [J, grad] = lrCostFunction(theta, X, y, lambda)
	hypotheses = sigmoid(X * theta);
	m = rows(X);
	J_t = (-y.*log(hypotheses)) - ((1 - y).*log(1 - hypotheses));
	%sum(J_t
	J = sum(J_t)/m + (lambda/(2*m)) * sum(theta .^ 2);
	grad = 1/m * X' * (hypotheses - y);
	grad = grad + (lambda / m) .* theta;
endfunction
