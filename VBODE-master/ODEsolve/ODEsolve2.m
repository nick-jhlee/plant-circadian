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
% 
% toc1p=[1, 5, 9, 13, 17, 21; ...
%    0.3318,  0.3314, 0.6081, 0.8149, 0.8149, 0.8149 ];
% ztotp=[1, 5, 9, 13, 17, 21; ...
%      0.7588,0.7589,  0.7410,0.7287, 0.7287, 0.7287];

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
init_days = 2;

% % duplicate data, iteration 10000, lr=0.2
t_t= 0.5371;	
k_f= 6.07;
k_tZd= 8.7;
k_tZl=4.69;
d_t= 0.4845;
t_z= 6.74;
d_Zd=12.73;
k_l= 3.82;
k_d= 4.02;
d_Zl= 4.54;
d_tZd= 0.0346 ;
d_tZl =4.43;


t_interval = [0 96];
%t_interval = [0 24]+24*init_days;
init_cond = [0.5 0.5 0.5 0.5 0.5];
%init_cond = [init_toc1 init_ztlp 0 0 0];
%solution
%[t,y] = ode45(@(t,y) odefcn(t,y,d_G,d_P) , t_interval , init_cond);
%[t,y] = ode45(@(t,y) odefcn2(t,y,d_T,d_Z) , t_interval , init_cond);
[t,y] = ode45(@(t,y) odefcn3(t,y,t_t, k_f, k_tZd, k_tZl, d_t, t_z, d_Zd, k_l, k_d, d_Zl, d_tZd, d_tZl) , t_interval , init_cond);
figure(2)
plot(t,y(:,1),'b',t,y(:,2),'r');
%plot(t,y(:,3),'g',t,y(:,4),'g');
hold on;
plot(toc1p(1,:)+24*init_days,toc1p(2,:),'b.', 'MarkerSize', 20);
plot(toc1p(1,:)+24*(init_days+1),toc1p(2,:),'b.', 'MarkerSize', 20);
plot(ztotp(1,:)+24*init_days,ztotp(2,:),'r.','MarkerSize', 20);
plot(ztotp(1,:)+24*(init_days+1),ztotp(2,:),'r.','MarkerSize', 20);
%hold off
