%% selectBaseAttributes: Select a given set of attributes to be bases
function [X, y] = selectBaseAttributes(raw_data)
	% Y are the leagues of the players
	y = raw_data(:, 2);
	% Attributes from another kaggle kernel
	% X = [raw_data(:, [14, 12, 13, 5, 6, 7])];
	% All attributes
	X = [raw_data(:, 3:columns(raw_data))];
end
