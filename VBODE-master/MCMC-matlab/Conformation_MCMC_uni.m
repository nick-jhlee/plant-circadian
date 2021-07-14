function [oitdata,oizdata,oigdata,oip3data,okir,okaa] = Conformation_MCMC_uni(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,kaa)

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

unc1=1.3;
unc2=0.005;
terror=unc1.*([0.0431, 0.039, 0.069, 0.043, 0, 0]'+unc2);
zerror=unc1.*([0.014, 0.014, 0.115, 0.02, 0.086, 0.086]'+unc2);
gerror=unc1.*([0.0214103, 0.0508533, 0.128269, 0.0498922, 0.0121651, 0.192175, 0.104515, 0.0925685, 0.0813823]'+unc2);
p3error=unc1.*([0.00703093, 0.017507, 0.0281341, 0.027253, 0, 0.0132593, 0.0297481, 0.0610875, 0.0522455]'+unc2);

for iii=1:length(kir)
    
    tpr=tir; dpr=dir; kpr=kir; bpr=bir; ubpr=ubir;
    
    ir=kir(iii);
    pr=100*rand;
    kpr(iii)=pr;
    
    tmp=num2cell(tpr); dmp=num2cell(dpr); kmp=num2cell(kpr); bmp=num2cell(bpr); ubmp=num2cell(ubpr);
    
    [tt tz tg tp]=deal(tmp{:});
    [dt dz1 dz2 dg dp]=deal(dmp{:});
    [kc1 kc2]=deal(kmp{:});
    [bb]=deal(bmp{:});
    [ubtz1 ubtz2 ubtg ubtp ubzg1 ...
        ubzg2 ubzp1 ubzp2 ubgp]=deal(ubmp{:});
    
    plevel = [];
    C2=0*ones(1,14);
    
    for j=1:days
        light = 1;
        tspan = 24*(j-1):1:24*(j-1)+12;
        [T2,C2] = ode15s('multi_degradation_ODE_v2',tspan,C2(end,:));
        if j==days
            plevel = [plevel; C2];
        end
        
        light = 0;
        tspan = 24*(j-1)+12:1:24*j;
        [T2,C2] = ode15s('multi_degradation_ODE_v2',tspan,C2(end,:));
        if j==days
            plevel=[plevel; C2(2:end,:)];
        end
    end
    
    ptdata=plevel(:,1)+plevel(:,6)+plevel(:,7)+plevel(:,8)+plevel(:,9);
    pzdata=plevel(:,2)+plevel(:,3)+plevel(:,6)+plevel(:,7)+plevel(:,10)+plevel(:,11)+plevel(:,12)+plevel(:,13);
    pgdata=plevel(:,4)+plevel(:,8)+plevel(:,10)+plevel(:,11)+plevel(:,14);
    pp3data=plevel(:,5)+plevel(:,9)+plevel(:,12)+plevel(:,13)+plevel(:,14);
    
    ptdata=ptdata(toc1p(1,:)+1,1); pzdata=pzdata(ztlp(1,:)+1,1);
    pgdata=pgdata(gip(1,:)+1,1); pp3data=pp3data(prr3p(1,:)+1,1);
    
    ptdata=ptdata./max(ptdata); pzdata=pzdata./max(pzdata);
    pgdata=pgdata./max(pgdata); pp3data=pp3data./max(pp3data);
    
    f1=prod([normpdf(toc1p(2,:)',itdata,terror); normpdf(ztlp(2,:)',izdata,zerror)...
        ; normpdf(gip(2,:)',igdata,gerror); normpdf(prr3p(2,:)',ip3data,p3error)]);
    f2=prod([normpdf(toc1p(2,:)',ptdata,terror); normpdf(ztlp(2,:)',pzdata,zerror)...
        ; normpdf(gip(2,:)',pgdata,gerror); normpdf(prr3p(2,:)',pp3data,p3error)]);
    
    mtan=1/4;
    
    prior1=(1-tanh(mtan*abs(log10(kir(1)/kir(2)))));
    prior2=(1-tanh(mtan*abs(log10(kpr(1)/kpr(2)))));  
    
    p1=1; p2=1;
    u=rand;
    
    if u<min([1 (prior2*f2*p1)/(prior1*f1*p2)])
        
        kir=kpr;
        
        kaa(iternum,iii)=1;
        itdata=ptdata; izdata=pzdata;
        igdata=pgdata; ip3data=pp3data;
        
    end
    
end

oitdata=itdata;
oizdata=izdata;
oigdata=igdata;
oip3data=ip3data;

okir=kir;
okaa=kaa;

end

