function [predictions] = predict(thetas, X)
  num_labels = columns(thetas);
  predictions = zeros(rows(X), num_labels);
  for i = 1:num_labels
    predictions(:, i) = sigmoid(X * thetas(:, i));
  endfor
end  % predict
