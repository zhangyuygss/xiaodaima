function fval = f(x)
	%fval = 10*(x(1)^2) + x(2)^2;
%    fval = x(1)^2 + x(2)^2;
    % fval = 2*x(1)^2 - 2*x(1)*x(2) + x(2)^2 + 2*x(1) - 2*x(2);
    fval = 0.5*x(1)^2*(x(1)^2/6 + 1) + x(2)*atan(x(2)) - 0.5*log(x(2)^2 + 1);
end