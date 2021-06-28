function dydt = odefcn2(t,y,d_T,d_Z)
% Y = [ T Z ]
toc1mrna=[0 1 5 9 13 17 21 24; ...
    0.401508 0.376 0.376 0.69 1 0.52 0.489 0.401508];
gimrna=[0 3 6 9 12 15 18 21 24; ...
    0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789];
prr3mrna=[0 3 6 9 12 15 18 21 24; ...
    0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];
dydt = zeros(2,1);
dydt(1) = interp1(toc1mrna(1,:),toc1mrna(2,:),mod(t,24)) -d_T * y(1);
dydt(2) =  1- d_Z * y(2);

end