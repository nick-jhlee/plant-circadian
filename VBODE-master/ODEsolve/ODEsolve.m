%parameters
toc1mrna=[0 1 5 9 13 17 21 24; ...
    0.401508 0.376 0.376 0.69 1 0.52 0.489 0.401508];
gimrna=[0 3 6 9 12 15 18 21 24; ...
    0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789];
prr3mrna=[0 3 6 9 12 15 18 21 24; ...
    0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];
toc1p=[1 5 9 13 17 21; ...
    0.0649 0.0346 0.29 0.987 1 0.645];
ztotp=[1, 5, 9, 13, 17, 21; ...
    0.115, 0.187, 0.445, 1., 0.718, 0.56];
gip=[0 3 6 9 12 15 18 21 24; ...
    0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325];
prr3p=[0 3 6 9 12 15 18 21 24; ...
    0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436];

init_gi =  0.237939;
init_prr3 =  0.021049;
init_toc1 = 0;
init_ztlp =  0;     
% init_gi =  0.222;
% init_prr3 =  0.0167;
d_T = 1.73117887973785;
d_Z = 0.942879080772399;
d_G = 0.853;
d_P = 0.819;
init_days = 1;
%time interval and initial condition
% 
% t_t= 0.727;
% k_f= 16.3;
% k_tZd= 4.44;
% k_tZl=4.64;
% d_t= 0.50613;
% t_z= 13.07508;
% d_Zd= 17.76;
% k_l= 4.78;
% k_d= 2.10955;
% d_Zl= 3.23;
% d_tZd= 0.00793;
% d_tZl = 5.16;


t_t= 50.46;
k_f= 17.5;
k_tZd= 70;
k_tZl=53;
d_t= 53.5;
t_z= 60.4;
d_Zd= 103.4;
k_l= 43.9;
k_d= 37.1;
d_Zl= 48.4;
d_tZd=11.2;
d_tZl =43.4;

t_interval = [0 72];
%t_interval = [0 24]+24*init_days;
init_cond = [init_toc1 init_ztlp 0 0 0];
%solution
%[t,y] = ode45(@(t,y) odefcn(t,y,d_G,d_P) , t_interval , init_cond);
%[t,y] = ode45(@(t,y) odefcn2(t,y,d_T,d_Z) , t_interval , init_cond);
[t,y] = ode45(@(t,y) odefcn3(t,y,t_t, k_f, k_tZd, k_tZl, d_t, t_z, d_Zd, k_l, k_d, d_Zl, d_tZd, d_tZl) , t_interval , init_cond);
figure(2)
plot(t,y(:,1),'b',t,y(:,2),'r');
hold on;
plot(toc1p(1,:)+24*init_days,toc1p(2,:),'b.', 'MarkerSize', 20);
plot(ztotp(1,:)+24*init_days,ztotp(2,:),'r.','MarkerSize', 20);
%hold off
