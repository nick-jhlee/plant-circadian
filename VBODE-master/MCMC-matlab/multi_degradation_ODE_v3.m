function dC = multi_degradation_ODE_v3(t,C)

global tt tz tg tp ... 
    dt dz1 dz2 dg dp ...
    dtz1 dtz2 dtg  dtp dz1g dz2g dz1p dz2p dgp ...
    kc1 kc2 ...
    bb ... 
    ubtz1 ubtz2 ubtg ubtp ubzg1 ubzg2 ubzp1 ubzp2 ubgp ...
    light

%  ODE function

dC=zeros(14,1);

%C(1): TOC1 protein C(2): ZTL protein (ZTL1) C(3): ZTL protein transfomred by light (ZTL2) 
%C(4): GI protein C(5): PRR3 protein  C(6): TOC1-ZTL1 complex C(7): TOC1-ZTL2 complex 
%C(8): TOC1-GI complex C(9): TOC1-PRR3 complex C(10): ZTL1-GI complex C(11): ZTL2-GI complex
%C(12): ZTL1-PRR3 complex C(13): ZTL2-PRR3 complex C(14): GI-PRR3 complex

% TOC1 protein
dC(1) = tt*toc1mrnainter(t) - dt*C(1) - bb*C(1)*C(2) - bb*C(1)*C(3) - bb*C(1)*C(4) - bb*C(1)*C(5) + ubtz1*C(6) + ubtz2*C(7) + ubtg*C(8) + ubtp*C(9);

% ZTL protein (ZTL1)
dC(2) = tz - dz1*C(2) - kc1*light*C(2) + kc2*(1-light)*C(3)  - bb*C(1)*C(2) - bb*C(2)*C(4) - bb*C(2)*C(5) + ubtz1*C(6) + ubzg1*C(10) + ubzp1*C(12);

% ZTL protein transfomred by light (ZTL2)
dC(3) = kc1*light*C(2) - kc2*(1-light)*C(3) - dz2*C(3) - bb*C(1)*C(3) - bb*C(3)*C(4) - bb*C(3)*C(5) + ubtz2*C(7) + ubzg2*C(11) + ubzp2*C(13);

% GI protein 
dC(4) = tg*gimrnainter(t) - dg*C(4) - bb*C(1)*C(4) - bb*C(2)*C(4) - bb*C(3)*C(4) - bb*C(4)*C(5) + ubtg*C(8) + ubzg1*C(10) + ubzg2*C(11) + ubgp*C(14);

% PRR3 protein
dC(5) = tp*prr3mrnainter(t) - dp*C(5) - bb*C(1)*C(5) - bb*C(2)*C(5) - bb*C(3)*C(5) - bb*C(4)*C(5) + ubtp*C(9) + ubzp1*C(12) + ubzp2*C(13) + ubgp*C(14);

% TOC1-ZTL1 complex
dC(6) = bb*C(1)*C(2) - ubtz1*C(6) - dtz1*C(6);

% TOC1-ZTL2 complex
dC(7) = bb*C(1)*C(3) - ubtz2*C(7) - dtz2*C(7);

% TOC1-GI protein
dC(8) = bb*C(1)*C(4) - ubtg*C(8) - dtg *C(8);

% TOC1-PRR3 protein
dC(9) = bb*C(1)*C(5) - ubtp*C(9) -  dtp*C(9);

% ZTL1-GI protein
dC(10) = bb*C(2)*C(4) - ubzg1*C(10) -  dz1g *C(10);

% ZTL2-GI protein
dC(11) = bb*C(3)*C(4) - ubzg2*C(11) - dz2g *C(11);

% ZTL1-PRR3 protein
dC(12) = bb*C(2)*C(5) - ubzp1*C(12) - dz1p *C(12);

% ZTL2-PRR3 protein
dC(13) = bb*C(3)*C(5) - ubzp2*C(13) - dz2p *C(13);

% GI-PRR3 protein
dC(14) = bb*C(4)*C(5) - ubgp*C(14) - dgp *C(14);


    %% TOC1 mrna interpolation
    function [outputArg1] = toc1mrnainter(inputArg1)
    global toc1mrna
    outputArg1=interp1(toc1mrna(1,:),toc1mrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

    %% GI mrna interpolation
    function [outputArg1] = gimrnainter(inputArg1)
    global gimrna
    outputArg1=interp1(gimrna(1,:),gimrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

    %% PRR3 mrna interpolation
    function [outputArg1] = prr3mrnainter(inputArg1)
    global prr3mrna
    outputArg1=interp1(prr3mrna(1,:),prr3mrna(2,:),mod(inputArg1,24));
    if outputArg1 < 0
        outputArg1=0;
    end
    end

end

