clear;
mu = 1.4845;
sigma = 0.54;
length = 1000;
std = randn(length, 1);
x = repmat(mu, length, 1) + std*sqrt(sigma);        % (a)
nbins = 60;
h = histogram(x, nbins);                            % (b)
figure;
x_mean = mean(x);
x_var = var(x);

correlation_test(1, 2, 1.5, 2.3, 0);
figure;
correlation_test(10, 1, 1.5, 2.3, 0.97);
figure;
correlation_test(1, 2, 1.5, 2.3, 0.7);

function correlation_test(x_mean, y_mean, sigma_x, sigma_y, rho)
    dlength = 1000;
    disp(['Input: Xmean ' num2str(x_mean) ' sigmaX ' num2str(sigma_x)]);
    disp(['       Ymean ' num2str(y_mean) ' sigmaY ' num2str(sigma_y) ' rho' num2str(rho)]);
    y = repmat(y_mean, dlength, 1) + randn(dlength, 1)*sqrt(sigma_y);
    yd_mean = mean(y);          % Y sample mean
    disp(['Y sample mean: ' num2str(yd_mean)]);
    yd_var = var(y);
    disp(['Y sample variance: ' num2str(yd_var)]);
    
    x = zeros(dlength, 1);
    xc_sigma = sigma_x*(1 - rho^2);     % Conditional distribution of X
    for iter = 1:dlength
       xc_mean = x_mean + rho*sqrt(sigma_x)*(y(iter) - y_mean)/sqrt(sigma_y);
       x(iter) = xc_mean + randn*sqrt(xc_sigma);
    end
    
    xd_mean = mean(x);
    disp(['X sample mean: ' num2str(xd_mean)]);
    xd_var = var(x);
    disp(['X sample mean: ' num2str(xd_var)]);
    
    % Sample rho
    rho_d = (x - xd_mean)'*(y - yd_mean)/((dlength-1)*sqrt(xd_var*yd_var));
    disp(['Sample rho:' num2str(rho_d)]);
    
    % Scatter diagram
    scatter(x(1:100), y(1:100));
end
