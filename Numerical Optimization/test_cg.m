clear;

% % convex1 test
% n = 10^4; 
% x0 = zeros(n, 1);
% for iter = 1:n
% 	x0(iter) = iter/n;
% end
% tic
% [xs, vals, k] = conjugate_gradient('fconvex1', 'gconvex1', x0, 'PRP');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fconvex1', 'gconvex1', x0, 'FR');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fconvex1', 'gconvex1', x0, 'PRP+');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fconvex1', 'gconvex1', x0, 'CD');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fconvex1', 'gconvex1', x0, 'DY');
% toc

% % convex2 test
% n = 10^4; 
% x0 = ones(n, 1);
% tic
% % [xs, vals, k] = conjugate_gradient('fconvex2', 'gconvex2', x0, 'PRP');
% toc
% tic
% % [xs, vals, k] = conjugate_gradient('fconvex2', 'gconvex2', x0, 'FR');
% toc
% tic
% % [xs, vals, k] = conjugate_gradient('fconvex2', 'gconvex2', x0, 'PRP+');
% toc
% tic
% % [xs, vals, k] = conjugate_gradient('fconvex2', 'gconvex2', x0, 'CD');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fconvex2', 'gconvex2', x0, 'DY');
% toc

% % Penalty test
% n = 10^4; 
% x0 = ones(n, 1);
% for iter = 1:n
% 	x0(iter) = iter;
% end
% tic
% % [xs, vals, k] = conjugate_gradient('fpenalty', 'gpenalty', x0, 'PRP');
% toc
% tic
% % [xs, vals, k] = conjugate_gradient('fpenalty', 'gpenalty', x0, 'FR');
% toc
% tic
% % [xs, vals, k] = conjugate_gradient('fpenalty', 'gpenalty', x0, 'PRP+');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fpenalty', 'gpenalty', x0, 'CD');
% toc
% tic
% [xs, vals, k] = conjugate_gradient('fpenalty', 'gpenalty', x0, 'DY');
% toc

% Trigonometric test
n = 10^4; 
x0 = ones(n, 1)/n;
tic
[xs, vals, k] = conjugate_gradient('ftrig', 'gtrig', x0, 'PRP');
toc
tic
[xs, vals, k] = conjugate_gradient('ftrig', 'gtrig', x0, 'FR');
toc
tic
[xs, vals, k] = conjugate_gradient('ftrig', 'gtrig', x0, 'PRP+');
toc
tic
[xs, vals, k] = conjugate_gradient('ftrig', 'gtrig', x0, 'CD');
toc
tic
[xs, vals, k] = conjugate_gradient('ftrig', 'gtrig', x0, 'DY');
toc
