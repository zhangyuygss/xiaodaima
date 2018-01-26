% %  A demo of Central Limit Theroy
clear;
% % Uniform distribution
% dx = 0.005;
% x = [0 : dx : 1-dx]';
% p_x = ones(length(x), 1);     
% p_xext = [p_x ; 1];          
% p_s = p_x;
% max_vr_num = 8;
% subplot(3,3,1);
% plot(x, p_x);
% 
% for iter = 1:max_vr_num
%     p_s = conv(p_s, p_xext);
%     p_s = p_s*0.005;
%     x = [0 : dx : (iter+1-dx)]';
%     subplot(3,3,iter+1);
%     plot(x, p_s);
%     pause;
% end


% % Gamma distribution
% dx = 0.00005;
% x = [0 : dx : 0.01-dx]';
% p_x = gamma(x);                 
% p_xext = [p_x; gamma(1)];         
% p_s = p_x;
% max_vr_num = 8;
% 
% subplot(3,3,1);
% plot(x, p_x);
% 
% for iter = 1:max_vr_num
%     p_s = conv(p_s, p_xext);
%     x = [0 : dx : (iter*0.01+0.01-dx)]';
%     subplot(3,3,iter+1);
%     plot(x, p_s);
%     pause;
% end
% 
% 

% % Bernoulli distribution
% dx = 0.005;
% x = [0 : dx : 1-dx]';
% p_x = zeros(length(x), 1); 
% p_x(1) = 0.5;
% p_x(length(x)) = 0.5;
% p_xext = [p_x ; 0];          
% p_s = p_x;
% max_vr_num = 8;
% subplot(3,3,1);
% plot(x, p_x);
% 
% for iter = 1:max_vr_num
%     p_s = conv(p_s, p_xext);
% %     p_s = p_s*0.5;
%     x = [0 : dx : (iter+1-dx)]';
%     subplot(3,3,iter+1);
%     plot(x, p_s);
%     pause;
% end

% Mixture
dx = 0.005;
x = [0 : dx : 1-dx]';
p_b = zeros(length(x), 1); 
p_b(1) = 0.5;
p_b(length(x)) = 0.5;
p_b1 = [p_b ; 0];          
p_s = p_b;
max_vr_num = 9;

p_u = ones(length(x), 1);
p_u1 = [p_u; 1];
p_u1 = p_u1*0.005;


for iter = 1:max_vr_num
    if(mod(iter, 2) == 1)
       p_s = conv(p_s, p_u1);
%        p_s = p_s*0.005;
    else
       p_s = conv(p_s, p_b1);
    end
    
%     p_s = p_s*0.5;
    x = [0 : dx : (iter+1-dx)]';
    subplot(3,3,iter);
    plot(x, p_s);
    pause;
end












