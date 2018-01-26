clear;
x0 = [0; 0];
[x, val, k, plotinfo] = steepest_descent('f', 'g', x0);

x1 = [-2:0.1:2]';
x2 = x1;
[x1, x2] = meshgrid(x1, x2);
y = 2*x1.^2 - 2*x1.*x2 + x2.^2 + 2*x1 - 2*x2;
surfc(x1, x2, y); hold on;
p = plot3(plotinfo(:,1), plotinfo(:,2), plotinfo(:,3), 'r');
p.Marker = '*';
