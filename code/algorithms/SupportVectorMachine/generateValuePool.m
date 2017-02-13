function [pool] = generateValuePool(seeds, iterations)
  pool = zeros(columns(seeds), iterations);
  for i = 0:iterations
    pool(:, i + 1) = seeds .* (10 ^ i);
  endfor
  % Vectorize pool
  poolsize = numel(pool);
  pool = reshape(pool, 1, poolsize);
endfunction
