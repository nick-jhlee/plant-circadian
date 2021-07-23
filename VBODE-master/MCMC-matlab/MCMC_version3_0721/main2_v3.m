function main2_v3(repeat_num)
% clear all;
% close all;
% clc;

cc = clock;
rng(cc(end));

global tt tz tg tp ... 
    dt dz1 dz2 dg dp ...
    dtz1_t dtz1_z1 dtz2_t dtz2_z2 dtg_t dtg_g dtp_t dtp_p dz1g_z1 dz1g_g dz1p_z1 dz1p_p dz2g_z2 dz2g_g dz2p_z2 dz2p_p dgp_g dgp_p ... %AB -> A : dAB_A*dB
    kc1 kc2 ...
    bb ... 
    ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2 ubgp ...
    light  days toc1mrna gimrna prr3mrna iternum 

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
% dmatrix=dec2bin(0:63)-'0';
% deci=dmatrix(fnum,:);
% csvwrite(['deci', num2str(fnum),'.csv'],deci);

iter=5000;
tscale=100;

for repeat=1:1:100
    
    tinit=[1 1 1 1]; dinit=[1 1 1 1 1];
    cdinit = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
    kinit=[1 1];
    binit=tscale;
    ubinit=tscale.*[1 1 1 1 1 1 1 1 1];
    
    tir=tinit; dir=dinit; kir=kinit; bir=binit; ubir=ubinit;
    cdir = cdinit;
    trr=tir; drr=dir; krr=kir; brr=bir; ubrr=ubir; cdrr=cdir;
    rr=[trr,drr,krr,brr,ubrr,cdrr];
    
    % Initialize acceptance ratio
    taa=zeros(iter,4); daa=zeros(iter,5);
    kaa=zeros(iter,2); baa=zeros(iter,1); ubaa=zeros(iter,9);
    cdaa=zeros(iter,18);
    aa=[taa,daa,kaa,baa,ubaa,cdaa];
    
    tmp=num2cell(tir); dmp=num2cell(dir); kmp=num2cell(kir);
    bmp=num2cell(bir); ubmp=num2cell(ubir); cdmp=num2cell(cdir);
    
    [tt tz tg tp]=deal(tmp{:});
    [dt dz1 dz2 dg dp]=deal(dmp{:});
    [kc1 kc2]=deal(kmp{:});
    [bb]=deal(bmp{:});
    [ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2 ubgp]=deal(ubmp{:});
    [dtz1_t dtz1_z1 dtz2_t dtz2_z2 dtg_t dtg_g dtp_t dtp_p dz1g_z1 dz1g_g dz1p_z1 dz1p_p dz2g_z2 dz2g_g dz2p_z2 dz2p_p dgp_g dgp_p]=deal(cdmp{:});
    
    days=6;
    plevel = [];
    C1=0*ones(1,14);
    
    for j=1:days
        light = 1;
        tspan = 24*(j-1):1:24*(j-1)+12;
        [T1,C1] = ode15s('multi_degradation_ODE_v4',tspan,C1(end,:));
        if j==days
            plevel = [plevel; C1];
        end
        
        light = 0;
        tspan = 24*(j-1)+12:1:24*j;
        [T1,C1] = ode15s('multi_degradation_ODE_v4',tspan,C1(end,:));
        if j==days
            plevel=[plevel; C1(2:end,:)];
        end
    end
    
    %C(1): TOC1 protein C(2): ZTL protein (ZTL1) C(3): ZTL protein transfomred by light (ZTL2)
    %C(4): GI protein C(5): PRR3 protein  C(6): TOC1-ZTL1 complex C(7): TOC1-ZTL2 complex
    %C(8): TOC1-GI complex C(9): TOC1-PRR3 complex C(10): ZTL1-GI complex C(11): ZTL2-GI complex
    %C(12): ZTL1-PRR3 complex C(13): ZTL2-PRR3 complex C(14): GI-PRR3 complex
    
    itdata=plevel(:,1)+plevel(:,6)+plevel(:,7)+plevel(:,8)+plevel(:,9);
    izdata=plevel(:,2)+plevel(:,3)+plevel(:,6)+plevel(:,7)+plevel(:,10)+plevel(:,11)+plevel(:,12)+plevel(:,13);
    igdata=plevel(:,4)+plevel(:,8)+plevel(:,10)+plevel(:,11)+plevel(:,14);
    ip3data=plevel(:,5)+plevel(:,9)+plevel(:,12)+plevel(:,13)+plevel(:,14);
    
    itdata=itdata(toc1p(1,:)+1,1);
    izdata=izdata(ztlp(1,:)+1,1);
    igdata=igdata(gip(1,:)+1,1);
    ip3data=ip3data(prr3p(1,:)+1,1);
    
    itdata=itdata./max(itdata);
    izdata=izdata./max(izdata);
    igdata=igdata./max(igdata);
    ip3data=ip3data./max(ip3data);
    
    itd=itdata;
    izd=izdata;
    igd=igdata;
    ip3d=ip3data;
    
    
    for i=1:iter
        
        iternum=i;
        
        %% Translation MCMC
        [itdata,izdata,igdata,ip3data,tir,taa]=Translation_MCMC(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,taa);
        
        %% Degradation MCMC
        [itdata,izdata,igdata,ip3data,dir,daa]=Degradation_MCMC(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,daa);
        
        %% Conformation MCMC
        [itdata,izdata,igdata,ip3data,kir,kaa]=Conformation_MCMC(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,kaa);
        
        %% Binding MCMC
        [itdata,izdata,igdata,ip3data,bir,baa]=Binding_MCMC(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,baa);
        
        %% Unbinding MCMC
        [itdata,izdata,igdata,ip3data,ubir,ubaa]=Unbinding_MCMC_ver3(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,ubaa);
        
        %% Complex Degradation MCMC
        [itdata,izdata,igdata,ip3data,cdir,cdaa]=Complex_Degradation_MCMC(itdata,izdata,igdata,ip3data,tir,dir,kir,bir,ubir,cdir,cdaa);
        
        trr=[trr;tir];
        drr=[drr;dir];
        krr=[krr;kir];
        brr=[brr;bir];
        ubrr=[ubrr;ubir];
        cdrr=[cdrr;cdir];
        rr=[trr,drr,krr,brr,ubrr,cdrr];
        aa=[taa,daa,kaa,baa,ubaa,cdaa];
        
        itd=[itd,itdata];
        izd=[izd,izdata];
        igd=[igd,igdata];
        ip3d=[ip3d,ip3data];
        
        %   csvwrite(['trr', num2str(fnum), '.csv'],trr);
        %   csvwrite(['drr', num2str(fnum), '.csv'],drr);
        %   csvwrite(['brr', num2str(fnum), '.csv'],brr);
        %   csvwrite(['ubrr', num2str(fnum), '.csv'],ubrr);
        %
        %   csvwrite(['rr', num2str(fnum),'_',num2str(repeat),'.csv'],rr);
        %   csvwrite(['aa', num2str(fnum),'_',num2str(repeat),'.csv'],aa);
        %   csvwrite(['itd', num2str(fnum),'_',num2str(repeat),'.csv' ],itd);
        %   csvwrite(['izd', num2str(fnum),'_',num2str(repeat),'.csv' ],izd);
        %   csvwrite(['igd' , num2str(fnum),'_',num2str(repeat),'.csv'],igd);
        %   csvwrite(['ip3d' , num2str(fnum),'_',num2str(repeat),'.csv'],ip3d);
        
        %   csvwrite(['rr_',num2str(repeat),'.csv'],rr);
        %   csvwrite(['aa_',num2str(repeat),'.csv'],aa);
        %   csvwrite(['itd_',num2str(repeat),'.csv' ],itd);
        %   csvwrite(['izd_',num2str(repeat),'.csv' ],izd);
        %   csvwrite(['igd_',num2str(repeat),'.csv'],igd);
        %   csvwrite(['ip3d_',num2str(repeat),'.csv'],ip3d);
       
        abs_error = sqrt( norm(toc1p(2,:)-itdata)^2 + norm(ztlp(2,:)-izdata)^2 + norm(gip(2,:)-igdata)^2 + norm(prr3p(2,:)-ip3data)^2 );
        
        writematrix(rr,['try_',num2str(repeat_num),'_rr_',num2str(repeat),'.csv']);
        writematrix(aa,['try_',num2str(repeat_num),'_aa_',num2str(repeat),'.csv']);
        writematrix(itd,['try_',num2str(repeat_num),'_itd_',num2str(repeat),'.csv']);
        writematrix(izd,['try_',num2str(repeat_num),'_izd_',num2str(repeat),'.csv']);
        writematrix(igd,['try_',num2str(repeat_num),'_igd_',num2str(repeat),'.csv']);
        writematrix(ip3d,['try_',num2str(repeat_num),'_ip3d_',num2str(repeat),'.csv']);
         writematrix(abs_error,['try_',num2str(repeat_num),'_MSE_',num2str(repeat),'.csv']);  
        
        if mod(iternum,2) == 0 % for print
            iternum
            abs_error
        end
    end
end 