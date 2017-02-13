function [hypothesis, a_1, z_2, a_2] = propagate(example, Theta1, Theta2)
	a_1 = [ones(rows(example), 1), example];
  z_2 = Theta1 * a_1';
	a_2 = sigmoid(z_2);
	a_2 = [ones(1, columns(a_2)); a_2];
  z_3 = Theta2 * a_2;
	hypothesis = sigmoid(z_3);
endfunction
