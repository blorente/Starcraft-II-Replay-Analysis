function [J, grad] = costNN(params_rn, num_attribs, nodos_ocultos, num_etiquetas, X, y, lambda)
  % Capturar las matrices de pesos originales
  Theta1 = reshape (params_rn(1:nodos_ocultos * (num_attribs + 1)), nodos_ocultos, (num_attribs + 1));
  Theta2 = reshape (params_rn((1 + (nodos_ocultos * (num_attribs + 1))):end), num_etiquetas, (nodos_ocultos + 1));

  % Extend matrix y
  y_unrolled = unrolly(y, num_etiquetas)';

  % Use front-propagation to calculate hypotheses
  a_1 = [ones(rows(X), 1), X];
  z_2 = Theta1 * a_1';
	a_2 = sigmoid(z_2);
	a_2 = [ones(1, columns(a_2)); a_2];
  z_3 = Theta2 * a_2;
	hypotheses = sigmoid(z_3);

  % Now hypotheses holds a hypothesis in each column,
  % with certainties for each label in a row

  % Calculate the cost for each label
  J = sum(sum(-y_unrolled.*log(hypotheses) - ((1 - y_unrolled).*log(1 - hypotheses))));
  % Adjust to the number of examples
  m = rows(X);
  J = (1 / m) * J;
  % Apply regularization parameter
  Theta1_trim = Theta1(:, 2:columns(Theta1));
  Theta2_trim = Theta2(:, 2:columns(Theta2));
  J = J + ((lambda/(2*m))*(sum(sum(Theta1_trim .* Theta1_trim)) + sum(sum(Theta2_trim .* Theta2_trim))));

  % Back propagation for the gradient

  % Calculate deviations from y
  deviation3 = hypotheses - y_unrolled;
  deviation2 = (Theta2' * deviation3) .* dsigmoid(z_2);
  Delta_2 = zeros(size(Theta2));
  Delta_1 = zeros(size(Theta1));

  % Calculate weight deltas
	Delta_2 = deviation3 * a_2';
	deviation2_trim = deviation2(2:size(deviation2,1), :);
 	Delta_1 = deviation2_trim * a_1;

  % Calculate gradient with regularization
  Theta1_grad = ((1 / m) .* Delta_1) + ((lambda/m) .* [zeros(nodos_ocultos, 1 ), Theta1(:,2:columns(Theta1))]);
  Theta2_grad = ((1 / m) .* Delta_2) + ((lambda/m) .* [zeros(num_etiquetas, 1 ), Theta2(:,2:columns(Theta2))]);
  % Roll the thetas into a gradient
  grad = [Theta1_grad(:); Theta2_grad(:)];
endfunction
