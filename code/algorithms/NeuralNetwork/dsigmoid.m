function R = dsigmoid(X)
  sig = sigmoid(X);
  sig_ext = [ones(1, columns(sig)); sig];
  R = sig_ext.*(1.-sig_ext);
endfunction
