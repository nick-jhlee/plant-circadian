%% Checking stiff ODE solver...

% Apply solver

tspan = [0, 24];    % Time period
x0 = 0; % Initial state
[t, x] = ode15s(@(t,x) model(t,x), tspan, x0);

% Plot
plot(t, x, '-o')
title('MATLAB')


% Define ODE model
function output = model(t_, x_)
    degradation = 1;    % degradation rate of x
    A = 1;
    theta = 2*pi/24;
    w = 0;
    output = A*(1 + cos(theta*t_ - w)) - degradation*x_;
end