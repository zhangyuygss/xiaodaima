clear; 

% x0 = [1, 0.7]';
% [x, val, k] = newton_basic('f', 'g', 'hesse', x0, 1);

% n = 15;
% % x0  =[0, 0, 0]';
% % [x, vals, k] = revised_newton('f', 'g', 'hesse', x0);
% % plotpath('f', x, vals);


% test differential equation
n = 3;
p = zeros(n*3, 1);
[x, vals, k] = quasi_newton('fnde', 'gnde', p, 'BFGS');

prst = x(end, :)';
v = prst(1:n);
theta = prst(n+1: 2*n);
omega = prst(2*n+1 : 3*n);

xp = [1:0.05:2];
yreal = xp.^4/5 + 1./(5*xp);

ysolve = zeros(1, length(xp));
for iter = 1:length(xp)
   xcur = xp(iter);
   ysolve(iter) = 2/5 + xcur*v'*sigmoid(xcur*omega - theta);
end
plot(xp, yreal, xp, ysolve, '--');

% % test watson
% n = 6;
% x0 = zeros(n, 1);
% [x, vals, k] = quasi_newton('fwatson', 'gwatson', x0, 'BFGS');

% % test discrete boundary value
% n = 15;
% h = 1/(n+1);
% is = 1:n;
% x0 = is*h;
% x0 = x0.*(x0-1);
% x0 = x0';
% [x, vals, k] = quasi_newton('fdbv', 'gdbv', x0, 'BFGS');
% 



