function [X_train, y_train, X_val, y_val, X_test, y_test] = splitSamples(X, y)
  m = rows(X);

  train_percent = 0.8;
  train_size = floor(train_percent * m);
  val_percent = 0.1;
  val_size = floor(val_percent * m);
  test_percent = 0.1;
  test_size = floor(test_percent * m);

  index_vector = randperm(m);
  index = 1;
  train_indices = index_vector(index:(index +train_size));
  index += train_size;
  val_indices = index_vector(index:(index + val_size));
  index += val_size;
  test_indices = index_vector(index:(index + test_size));

  X_train = X(train_indices, :);
  y_train = y(train_indices, :);
  X_val = X(val_indices, :);
  y_val = y(val_indices, :);
  X_test = X(test_indices, :);
  y_test = y(test_indices, :);
end  % splitSamples
