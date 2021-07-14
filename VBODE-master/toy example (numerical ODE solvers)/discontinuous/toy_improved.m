%% Checking stiff ODE solver...

% Apply solver

tspan_1 = [0, 12];    % Time period
tspan_2 = [12, 24];    % Time period
x0 = 0; % Initial state
[t_1, x_1] = ode15s(@(t,x) model_1(t,x), tspan_1, x0);
[t_2, x_2] = ode15s(@(t,x) model_2(t,x), tspan_2, x_1(end));

% Plot
% plot(t_1, x_1, '-o')
plot(t_1, x_1, '-o',  t_2, x_2, '-o')
title('MATLAB')


% Define ODE model
% For t < 12
function output = model_1(t_, x_)
    degradation = 1;    % degradation rate of x
    A = 1;
    theta = 2*pi/24;
    w = 0;
    output = A*(1 + cos(theta*t_ - w)) - degradation*x_ + x_;
end

% For t > 12
function output = model_2(t_, x_)
    degradation = 1;    % degradation rate of x
    A = 1;
    theta = 2*pi/24;
    w = 0;
    output = A*(1 + cos(theta*t_ - w)) - degradation*x_ + x_^2/11;
end