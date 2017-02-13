function [ Theta1, Theta2 ] = recoverThetas(Thetas, input_layer_size, hidden_layer_size, num_labels)
  Theta1 = reshape(Thetas(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(Thetas(1 + (hidden_layer_size * (input_layer_size + 1)):end), num_labels, hidden_layer_size + 1);
end  % recoverThetas
