function [correct, ratio] = applySVM(X_train, y_train, X_val, y_val, seeds, iterations, print_graph)
  pool = generateValuePool(seeds, iterations);
  printf('\n');
  printf('-------------------------------\n');
  printf('Applying Support Vector Machine\n');
  printf('-------------------------------\n');
  printf('Seeds: '); disp(seeds);
  printf('Iterations: %i\n', iterations);
  printf('\n');
  [correct, ratio] = generateModelsFromPool(X_train, y_train, X_val, y_val, pool, 'linear', print_graph);
end
