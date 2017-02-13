function [ Thetas ] = createInitialThetas(input_layer_size, hidden_layer_size, num_labels)
  Theta1 = randomWeights(input_layer_size, hidden_layer_size);
  Theta2 = randomWeights(hidden_layer_size, num_labels);

  Thetas = [Theta1(:); Theta2(:)];
end  % function
