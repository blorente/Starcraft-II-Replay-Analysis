function [ error_tr, error_val ] = getLearningCurves( X, y, Xval, Yval, num_labels, lambda, iterations )
  % Initial values
  m = rows(X)
  error_tr = zeros(m, 1);
  error_val = zeros(m, 1);

  iterations

  for i = 1:m
    % Generate subset of data to train with
    Xtrain = X(1:i, :);
    Ytrain = y(1:i);

    % Train the system to predict with that subset
    Theta_part = oneVsAll(Xtrain, Ytrain, num_labels, lambda, iterations);

    % Calculate the error rate for the validation data and training
    error_tr(i) = lrCostFunction(Theta_part, Xtrain, Ytrain, lambda)(1);
    error_val(i) = lrCostFunction(Theta_part, Xval, Yval, lambda)(1);
  endfor
endfunction
