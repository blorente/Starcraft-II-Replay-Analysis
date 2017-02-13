clear all;
warning('off');

% Import functions
source('./algorithms/LogisticRegression/manifest.m');
source('./algorithms/NeuralNetwork/manifest.m');
source('./algorithms/SupportVectorMachine/manifest.m');

raw_data = dlmread('rawdata/starcraft_bin.csv', ',');
% Ignore the first row (names)
raw_data = raw_data(2:rows(raw_data), :);
% Select base attributes
[X, y] = selectBaseAttributes(raw_data);
[X_train, y_train, X_val, y_val, X_test, y_test] = splitSamples(X, y);
num_labels = 2;

% LR parameters
iterations_LR = 5000;
lambdas_LR = 0:0.01:0.5;
graph_LR = true;
get_curves = false;

[correct_lr, ratio_lr] = applyLR(get_curves, X_train, y_train, X_val, y_val, num_labels, lambdas_LR, iterations_LR, graph_LR);

displayResults('LR', correct_lr, ratio_lr);
waitPress();

% NN parameters
input_layer_size = columns(X_train);
hidden_layer_sizes = [5, 10, 25, 50, 100, 150];
lambdas_nn = 0:0.1:5;
iterations_NN = 500;
graph_nn = true;

[correct_nn, ratio_nn] = applyNN(X_train, y_train, X_test, y_test, input_layer_size, hidden_layer_sizes, num_labels, lambdas_nn, iterations_NN, graph_nn);

displayResults('NN', correct_nn, ratio_nn);
waitPress();

% SVM parameters
CSigma_seeds = [0.01, 0.02, 0.03, 0.04];
CSigma_iterations = 5;
print_graph = true;
[correct_svm, ratio_svm] = applySVM(X_train, y_train, X_val, y_val, CSigma_seeds, CSigma_iterations, print_graph);

displayResults('SVM', correct_svm, ratio_svm);
waitPress();
