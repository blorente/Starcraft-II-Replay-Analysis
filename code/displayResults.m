function [] = displayResults(name, correct, ratios)
  printf('Correct %s:\n', name);
  disp(correct);
  printf('%s ratios:\n', name);
  disp(ratios .* 100);
end  % displayResults
