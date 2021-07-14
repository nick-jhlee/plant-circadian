% t<12
function dydt = odefcn3_1(t,y,t_t, k_f, k_tZd, k_tZl, d_t, t_z, d_Zd, k_l, k_d, d_Zl, d_tZd, d_tZl)
    %  y = [T, Ztot, Zd, TZd, TZl]
    toc1mrna=[0 1 5 9 13 17 21 24; ...
        0.401508 0.376 0.376 0.69 1 0.52 0.489 0.401508];
    gimrna=[0 3 6 9 12 15 18 21 24; ...
        0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789];
    prr3mrna=[0 3 6 9 12 15 18 21 24; ...
        0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];
    dydt = zeros(5,1);
    light=0;
    t_ = rem(t,24);
    if t_<=12
        light = 1;
    end

    % Zl was y(2) now (y(2)-y(3))

    dydt(1) = t_t *interp1(toc1mrna(1,:),toc1mrna(2,:),mod(t,24),'spline') - k_f * y(1) * y(2) + k_tZd * y(4)+ k_tZl * y(5) - d_t * y(1);
    dydt(2) =  t_z - k_f * y(1) * y(2) + k_tZd * y(4) + k_tZl * y(5)  - d_Zl *  (y(2)-y(3)) - d_Zd * y(3) ;
    dydt(3) = t_z - k_f * y(1) *y(3) + k_tZd * y(4) - d_Zd * y(3) - k_l * y(3);
    dydt(4) = k_f * y(1) * y(3) - k_tZd * y(4) - d_tZd * y(4);
    dydt(5) = k_f * y(1) *  (y(2)-y(3)) - k_tZl * y(5) - d_tZl * y(5);
end