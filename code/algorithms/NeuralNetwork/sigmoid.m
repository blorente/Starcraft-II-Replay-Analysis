function R = sigmoid(X)
  R = 1 ./ (1 .+ e.^(-X));
endfunction
