function grad = g(x)
	%grad = [20*x(1); 2*x(2)];
 %   grad = [2*x(1); 2*x(2)];
    % grad = [4*x(1) - 2*x(2) + 2 ; -2*x(1) + 2*x(2) - 2];
    grad = [x(1)^3/3 + x(1); atan(x(2))];
end
