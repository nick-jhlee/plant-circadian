toc1mrna=[0 1 5 9 13 17 21 24; ...
    0.401508 0.376 0.376 0.69 1 0.52 0.489 0.401508];
gimrna=[0 3 6 9 12 15 18 21 24; ...
    0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789];
prr3mrna=[0 3 6 9 12 15 18 21 24; ...
    0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];


xx = 0:.25:24;
figure(1)
yy1 = pchip(toc1mrna(1,:),toc1mrna(2,:),xx);
yy2 = makima(toc1mrna(1,:),toc1mrna(2,:),xx);
plot(toc1mrna(1,:),toc1mrna(2,:),'o',xx,yy1,'r',xx,yy2,'b')

figure(2)
yy1 = max(pchip(gimrna(1,:),gimrna(2,:),xx),0);
yy2 = max(makima(gimrna(1,:),gimrna(2,:),xx),0);
plot(gimrna(1,:),gimrna(2,:),'o',xx,yy1,'r',xx,yy2,'b')

figure(3)
yy1 = prr3mrnainter(xx);
yy2= max(makima(prr3mrna(1,:),prr3mrna(2,:),xx),0);
plot(prr3mrna(1,:),prr3mrna(2,:),'o',xx,yy1,'r',xx,yy2,'b')


function [outputArg1] = prr3mrnainter(inputArg1)
prr3mrna=[0 3 6 9 12 15 18 21 24; ...
    0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205];
outputArg1=max(0, pchip(prr3mrna(1,:),prr3mrna(2,:),mod(inputArg1,24)));
end

   