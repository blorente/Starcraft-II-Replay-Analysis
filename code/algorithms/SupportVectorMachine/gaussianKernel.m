function dist = gaussianKernel(x1, x2, sigma)
  dist = exp(-(sum((x1 - x2) .^ 2)) / (2 * sigma ^ 2));
endfunction
