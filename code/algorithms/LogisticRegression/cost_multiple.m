function [J, grad] = cost_multiple(theta, X, y, lambda)
  [J_base, grad_base] = cost(theta, X, y);
  J = J_base + (lambda /  2 * rows(X)) * sum(theta);
  grad_thetas = (lambda / rows(X)) .* theta;
  grad_thetas(1, 1) = 0;
  grad = grad_base + grad_thetas;
endfunction
