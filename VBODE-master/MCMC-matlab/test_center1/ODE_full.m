function dC = ODE_full(t,C,light,M)
% AB -> A, AB -> B but not AB -> O
global tt tz tg tp ... 
    dt dz1 dz2 dg dp ...
    dtz1_t dtz1_z1 dtz2_t dtz2_z2 dtg_t dtg_g dtp_t dtp_p dz1g_z1 dz1g_g dz1p_z1 dz1p_p dz2g_z2 dz2g_g dz2p_z2 dz2p_p dgp_g dgp_p ... %AB -> A : dAB_A*dB
    kc1 kc2 ...
    bb ... 
    ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2 ubgp ...
    days iternum 

N=num2cell(M);
[tt tz tg tp dt dz1 dz2 dg dp kc1 kc2 bb ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2...
    ubgp  dtz1_t dtz1_z1 dtz2_t dtz2_z2 dtg_t dtg_g dtp_t dtp_p dz1g_z1 dz1g_g dz1p_z1 dz1p_p dz2g_z2 dz2g_g dz2p_z2 dz2p_p dgp_g dgp_p] = deal(N{:});
%  ODE function
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

dC=zeros(14,1);

%C(1): TOC1 protein C(2): ZTL protein (ZTL1) C(3): ZTL protein transfomred by light (ZTL2) 
%C(4): GI protein C(5): PRR3 protein  C(6): TOC1-ZTL1 complex C(7): TOC1-ZTL2 complex 
%C(8): TOC1-GI complex C(9): TOC1-PRR3 complex C(10): ZTL1-GI complex C(11): ZTL2-GI complex
%C(12): ZTL1-PRR3 complex C(13): ZTL2-PRR3 complex C(14): GI-PRR3 complex

% TOC1 protein
dC(1) = tt*toc1mrnainter(t) - dt*C(1) - bb*C(1)*C(2) - bb*C(1)*C(3) - bb*C(1)*C(4) - bb*C(1)*C(5) + ubtz1*C(6) + ubtz2*C(7) + ubtg*C(8) +ubtp*C(9)+...
    dtz1_t*dz1* C(6)+dtz2_t*dz2* C(7)+ dtg_t*dg *C(8)+dtp_t*dp*C(9);

% ZTL protein (ZTL1)
dC(2) = tz - dz1*C(2) - kc1*light*C(2) + kc2*(1-light)*C(3)  - bb*C(1)*C(2) - bb*C(2)*C(4) - bb*C(2)*C(5) + ubtz1*C(6) + ubzg1*C(10) + ubzp1*C(12)+ ...
    dtz1_z1*dt* C(6)+dz1g_z1*dg*C(10)+ dz1p_z1 * dp* C(12);

% ZTL protein transfomred by light (ZTL2)
dC(3) = kc1*light*C(2) - kc2*(1-light)*C(3) - dz2*C(3) - bb*C(1)*C(3) - bb*C(3)*C(4) - bb*C(3)*C(5) + ubtz2*C(7) + ubzg2*C(11) + ubzp2*C(13)+ ...
    dtz2_z2*dt* C(7)+dz2g_z2*dg*C(11)+dz2p_z2 * dp * C(13);

% GI protein 
dC(4) = tg*gimrnainter(t) - dg*C(4) - bb*C(1)*C(4) - bb*C(2)*C(4) - bb*C(3)*C(4) - bb*C(4)*C(5) + ubtg*C(8) + ubzg1*C(10) + ubzg2*C(11) + ubgp*C(14)+ ...
    dtg_g*dt *C(8)+ dz1g_g*dz1*C(10)+dz2g_g * dz2 *C(11)+ dgp_g * dp * C(14);

% PRR3 protein
dC(5) = tp*prr3mrnainter(t) - dp*C(5) - bb*C(1)*C(5) - bb*C(2)*C(5) - bb*C(3)*C(5) - bb*C(4)*C(5) + ubtp*C(9) + ubzp1*C(12) + ubzp2*C(13) + ubgp*C(14)+ ...
    dtp_t*dt*C(9)+dz1p_z1 * dz1* C(12)+  dz2p_z2 * dz2 * C(13)+dgp_g * dg * C(14);

% TOC1-ZTL1 complex
dC(6) = bb*C(1)*C(2) - ubtz1*C(6) - dtz1_t*dt* C(6) - dtz1_z1*dz1* C(6);

% TOC1-ZTL2 complex
dC(7) = bb*C(1)*C(3) - ubtz2*C(7) - dtz2_t*dt* C(7) -dtz2_z2*dz2* C(7) ;

% TOC1-GI protein
dC(8) = bb*C(1)*C(4) - ubtg*C(8) - dtg_t*dt *C(8) - dtg_g*dg *C(8);

% TOC1-PRR3 protein
dC(9) = bb*C(1)*C(5) - ubtp*C(9) -  dtp_t*dt*C(9)-dtp_p*dp*C(9);

% ZTL1-GI protein
dC(10) = bb*C(2)*C(4) - ubzg1*C(10) -  dz1g_z1*dz1*C(10) - dz1g_g*dg*C(10);

% ZTL2-GI protein
dC(11) = bb*C(3)*C(4) - ubzg2*C(11) - dz2g_z2 * dz2 *C(11) - dz2g_g*dg*C(11);

% ZTL1-PRR3 protein
dC(12) = bb*C(2)*C(5) - ubzp1*C(12) - dz1p_z1 * dz1* C(12)-dz1p_p * dp* C(12);

% ZTL2-PRR3 protein
dC(13) = bb*C(3)*C(5) - ubzp2*C(13) - dz2p_z2 * dz2 * C(13) -dz2p_p * dp * C(13)  ;

% GI-PRR3 protein
dC(14) = bb*C(4)*C(5) - ubgp*C(14) - dgp_g * dg * C(14) -  dgp_p * dp * C(14);


    %% TOC1 mrna interpolation
    function [outputArg1] = toc1mrnainter(inputArg1)
    outputArg1=interp1(toc1mrna(1,:),toc1mrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

    %% GI mrna interpolation
    function [outputArg1] = gimrnainter(inputArg1)
    outputArg1=interp1(gimrna(1,:),gimrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

    %% PRR3 mrna interpolation
    function [outputArg1] = prr3mrnainter(inputArg1)
    outputArg1=interp1(prr3mrna(1,:),prr3mrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

end

