function [] = printCurve(X, y, error_train, error_val, lambda, fig )
  figure(fig);
  plot(1:rows(X), error_train, 1:rows(X), error_val);
  title(['Learning curve for linear regression. ', 'Lambda = ', num2str(lambda)])
  xlabel('Number of training examples')
  ylabel('Error')
endfunction
