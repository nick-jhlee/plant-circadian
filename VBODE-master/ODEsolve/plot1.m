t = linspace(0,24);
T=0.7;
dawn = 0;
dusk = 12;
y = 1/2 * ((1+tanh((t-24*floor(t/24)-dawn)/T)) -  (1+tanh((t-24*floor(t/24)-dusk)/T)) + (1+tanh((t-24*floor(t/24)-24)/T))    ) ;


plot(t,y)
