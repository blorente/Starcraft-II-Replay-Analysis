function W = randomWeights(in_nodes, out_nodes)
  eps_ini = sqrt(6) / sqrt(in_nodes + out_nodes);
  % Generar valores aleatorios entre 0 y 1
  W = rand(out_nodes, in_nodes + 1);
  % Ajustar al rango [-eini, eini]
  W = -eps_ini + (2*eps_ini) .* W;
endfunction
