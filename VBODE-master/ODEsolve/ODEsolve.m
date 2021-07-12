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
init_toc1 = 0.2;
init_ztlp =  0.51;     
% init_gi =  0.222;
% init_prr3 =  0.0167;
% d_T = 1.73117887973785;
% d_Z = 0.942879080772399;
% d_G = 0.853;
% d_P = 0.819;
init_days = 7;

t_t=1;
k_f= 10;
k_tZd= 1; % 내린다  / min diff
k_tZl= 0.01; %
d_t= 1; 
t_z= 1	; %compared max
d_Zd= 1; 
k_l =100 ; %redamp
k_d= 100; %
d_Zl= 10; %max
d_tZd=1;
d_tZl= 10;

d_G=.01;
t_interval = [0 24];
t_interval = [(init_days-1)*24 (init_days+2)*24+1];
%t_interval = [0 24]+24*init_days;
init_cond = [init_toc1 init_ztlp 0 0 0];
%solution
%[t,y] = ode45(@(t,y) odefcn(t,y,d_G,d_P) , t_interval , init_cond);
%[t,y] = ode45(@(t,y) odefcn2(t,y,d_T,d_Z) , t_interval , init_cond);
[t,y] = ode15s(@(t,y) odefcn3(t,y,t_t, k_f, k_tZd, k_tZl, d_t, t_z, d_Zd, k_l, k_d, d_Zl, d_tZd, d_tZl) , t_interval , init_cond);
%[t,y] = ode45(@(t,y) odefcnGI(t,y,d_G) , t_interval , [0.3]);
figure(1)
h1=plot(t,y(:,1)+y(:,3)+y(:,4),'b');
hold on;
% plot(t,y(:,1),'r');
% plot(t,y(:,4),'g');
% plot(t,y(:,5));
h2=plot(t,y(:,2)+y(:,4)+y(:,5),'r');
% h3=plot(t,y(:,1));
% h4=plot(t,y(:,2));
% h5=plot(t,y(:,3));
% h6=plot(t,y(:,4));
% h7=plot(t,y(:,5));
plot(toc1p(1,:)+24*(init_days),toc1p(2,:),'b.', 'MarkerSize', 20);
plot(toc1p(1,:)+24*(init_days+1),toc1p(2,:),'b.', 'MarkerSize', 20);
plot(ztotp(1,:)+24*init_days,ztotp(2,:),'r.','MarkerSize', 20);
plot(ztotp(1,:)+24*(init_days+1),ztotp(2,:),'r.','MarkerSize', 20);
% figure(3)
% t=1:1:24;
% y1=interp1(gimrna(1,:),gimrna(2,:),mod(t,24));
% plot(t,y1)
% y2 = interp1(toc1p(1,:),toc1p(2,:),mod(t,24));
% plot(t,y2)
%legend([h1 h2 h3 h4 h5 h6 h7],{'TOC1','ZTLtot', 'T', 'Zd', 'Zl', 'TZd', 'TZl'});
%hold off


