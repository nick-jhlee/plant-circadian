%parameters
toc1mrna=[0 1 5 9 13 17 21 24; ...
    0.401508 0.376 0.376 0.69 1 0.52 0.489 0.401508];
gimrna=[0 3 6 9 12 15 18 21 24; ...
    0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789];
prr3mrna=[0 3 6 9 12 15 18 21 24; ...
    0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];
toc1p=[1 5 9 13 17 21; ...
    0.0649 0.0346 0.29 0.987 1 0.645];
ztlp=[1, 5, 9, 13, 17, 21; ...
    0.115, 0.187, 0.445, 1., 0.718, 0.56];
gip=[0 3 6 9 12 15 18 21 24; ...
    0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325];
prr3p=[0 3 6 9 12 15 18 21 24; ...
    0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436];

init_gi =  0.237939;
init_prr3 =  0.021049;
d_G = 0.76;
d_P = 1.1;
%time interval and initial condition


t_interval = [0 24]+24*2;
init_cond = [init_gi init_prr3];
%solution
[t,y] = ode45(@(t,y) odefcn(t,y,d_G,d_P) , t_interval , init_cond);
%plot
%plot(t,y(:,1),'b',t,y(:,2),'r');
plot(t,y(:,1),'b',t,y(:,2),'r');
hold on;
plot(gip(1,:)+24*2,gip(2,:),'.', 'MarkerSize',20);
plot(prr3p(1,:)+24*2,prr3p(2,:),'.','MarkerSize',15);
hold off
