function [J, grad] = cost(theta, X, y)
  extended_theta = repmat(theta, rows(X), 1);
  hypotheses = sigmoid(dot(extended_theta, X, 2));
  positive_hypotheses = y.*log(hypotheses);
  negative_hypotheses = (1 - y).*log(1 - hypotheses);
  J = 1/rows(X) * sum(-positive_hypotheses - negative_hypotheses);
  grad = arrayfun(@(col_index) (1/rows(X) * sum((hypotheses - y) .* X(:, col_index))), 1:columns(X));
endfunction
