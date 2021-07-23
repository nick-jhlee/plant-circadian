
epsilon=1;
days=6;
plevel = [];
C1=0*ones(1,14);
M = readmatrix('result_param.csv');
toc1p=[1 5 9 13 17 21; ...
    0.0649 0.0346 0.29 0.987 1 0.645];
ztlp=[1, 5, 9, 13, 17, 21; ...
    0.115, 0.187, 0.445, 1., 0.718, 0.56];
gip=[0 3 6 9 12 15 18 21 24; ...
    0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325];
prr3p=[0 3 6 9 12 15 18 21 24; ...
    0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436];

for j=1:days
   tspan = 24*(j-1):1:24*(j-1)+12;
   [T1,C1] = ode15s(@(t,C) ODE_full(t, C, 1, M),tspan,C1(end,:));
   if j==days
       plevel = [plevel; C1];
   end
   
   tspan = 24*(j-1)+12:1:24*j;
   [T1,C1] = ode15s(@(t,C) ODE_full(t, C, 0, M),tspan,C1(end,:));
   if j==days
       plevel=[plevel; C1(2:end,:)];   
   end 
end

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

figure(1)
plot(toc1p(1,:),itdata, '-o', 'DisplayName','TOC1sim')
hold on;
plot(toc1p(1,:),toc1p(2,:), '-o', 'DisplayName','TOC1data')
legend
hold off;

figure(2)
plot(ztlp(1,:),izdata, '-o', 'DisplayName','ZTLsim')
hold on;
plot(ztlp(1,:),ztlp(2,:), '-o', 'DisplayName','ZTLdata')
legend
hold off;

figure(3)
plot(gip(1,:),igdata, '-o', 'DisplayName','GIsim')
hold on;
plot(gip(1,:),gip(2,:), '-o', 'DisplayName','GIdata')
legend
hold off;

figure(4)
plot(prr3p(1,:),ip3data, '-o', 'DisplayName','PRR3sim')
hold on;
plot(prr3p(1,:),prr3p(2,:), '-o', 'DisplayName','PRR3data')
legend
hold off;

abs_error = sqrt( norm(toc1p(2,:)-itdata)^2 + norm(ztlp(2,:)-izdata)^2 + norm(gip(2,:)-igdata)^2 + norm(prr3p(2,:)-ip3data)^2 ) 

% plot the good fitting
x = logspace(-5,5);
A = [];
A = [A, M(12:39)' ];
result = zeros(9,1);
% for  k=1:9
%     figure(10+k)
%     loglog(A(11+k)/A(1), A(10+2*k-1),'r.','MarkerSize',12);
%     hold on;    
%     loglog(A(11+k)/A(1),A(10+2*k),'b.','MarkerSize',12);
%     loglog(x, 2.^x * 0+1, 'k-','MarkerSize',12);
%     loglog(2.^x * 0+1, x,'k-','MarkerSize',12);
%     hold off;
% end