clear all;
close all;
clc;

fnum=1;
cc = clock;
rng(cc(end));

global tt tz tg tp ... 
    dt dz1 dz2 dg dp ...
    kc1 kc2 ...
    bb ... 
    ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2 ubgp ...
    light deci days toc1mrna gimrna prr3mrna iternum

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

terror=[0.0431, 0.039, 0.069, 0.043, 0, 0]';
zerror=[0.014, 0.014, 0.115, 0.02, 0.086, 0.086]';
gerror=[0.0214103, 0.0508533, 0.128269, 0.0498922, 0.0121651, 0.192175, 0.104515, 0.0925685, 0.0813823]';
p3error=[0.00703093, 0.017507, 0.0281341, 0.027253, 0, 0.0132593, 0.0297481, 0.0610875, 0.0522455]';

% Decision

dmatrix=dec2bin(0:63)-'0';
deci=dmatrix(fnum,:);
% csvwrite(['deci', num2str(fnum),'.csv'],deci);

iter=30000;
tscale=100;

tinit=[1 1 1 1]; dinit=[1 1 1 1 1];
kinit=[1 1];
binit=tscale; 
ubinit=tscale.*[1 1 1 1 1 1 1 1 1];

tir=tinit; dir=dinit; kir=kinit; bir=binit; ubir=ubinit;
trr=tir; drr=dir; krr=kir; brr=bir; ubrr=ubir;
rr=[trr,drr,krr,brr,ubrr];

% Initialize acceptance ratio
taa=zeros(iter,4); daa=zeros(iter,11);
kaa=zeros(iter,2); baa=zeros(iter,1); ubaa=zeros(iter,9);
aa=[taa,daa,kaa,baa,ubaa];


tmp=num2cell(tir); dmp=num2cell(dir); kmp=num2cell(kir); 
bmp=num2cell(bir); ubmp=num2cell(ubir);


[tt tz tg tp]=deal(tmp{:});
[dt dz1 dz2 dg dp]=deal(dmp{:});
[kc1 kc2]=deal(kmp{:});
[bb]=deal(bmp{:});
[ubtz1 ubtz2 ubtg ubtp ubzg1 ...
    ubzg2 ubzp1 ubzp2 ubgp]=deal(ubmp{:});


days=6;
plevel = [];
C1=0*ones(1,14);

for j=1:days
   light = 1;
   tspan = 24*(j-1):1:24*(j-1)+12;
   [T1,C1] = ode15s('multi_degradation_ODE_v2',tspan,C1(end,:));
   if j==days
   plevel = [plevel; C1];
   end
   
   light = 0;
   tspan = 24*(j-1)+12:1:24*j;
   [T1,C1] = ode15s('multi_degradation_ODE_v2',tspan,C1(end,:));
   if j==days
   plevel=[plevel; C1(2:end,:)];   
   end 
end


% plot(plevel(:,1),'b-'); hold on;
% plot(plevel(:,2),'r-'); hold off;
% xticks([0 12 24 36 48 60 72 84 96])
% % plot(plevel(:,1)+plevel(:,2)); hold off;




