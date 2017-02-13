function[ y_unrolled ] = unrolly(y, num_etiquetas)
  % Transformaciones de entrada para usar en funciones
  y_unrolled = zeros(rows(y), num_etiquetas);
  % Por cada valor de y, marcarlo en y_unrolled
  for i = 1:num_etiquetas
    marks = y == i;
    y_unrolled(:, i) = marks;
  endfor

endfunction
