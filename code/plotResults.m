function [] = plotResults(ratios, lambdas, name)
  disp(lambdas)
  h = plot(lambdas, ratios);
  title('Prediction accuracy');
  xlabel('Lambdas');
  ylabel('Prediction accuracies');
  print(name, '-dpng');
end
