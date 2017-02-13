function [hypothesis_matrix, a1, z2, a2] = hypothesis(X, Theta1, Theta2)
  hypothesis_matrix = ones(rows(X), rows(Theta2));
  for i = 1:rows(X)
    [hypothesis_matrix(i, :), a1(i,:), z2(i,:), a2(i,:)] = propagate(X(i, :), Theta1, Theta2);
    fprintf('Hypothesis %4i\r', i);
    if exist('OCTAVE_VERSION')
      fflush(stdout);
    end
  endfor
  fprintf('\n');
endfunction
