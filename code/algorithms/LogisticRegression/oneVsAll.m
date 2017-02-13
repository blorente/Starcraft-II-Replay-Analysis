function [all_theta] = oneVsAll(X, y, num_labels, lambda, iterations)
	% X ya ha sido extendida
	initial_theta = zeros(columns(X), 1);
	options = optimset('GradObj', 'on', 'MaxIter', iterations);

	all_theta = zeros(columns(X),  num_labels);

	for i = 1:num_labels
		all_theta(:, i) = fmincg(@(t) (lrCostFunction(t, X, (y == i), lambda)), initial_theta, options);
	endfor

endfunction
