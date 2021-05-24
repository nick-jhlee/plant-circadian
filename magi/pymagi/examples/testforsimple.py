import numpy as np
#from arma import ode_system, solve_magi
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

times1 = np.array([0, 1, 5, 9, 13, 17, 21, 24])
mTOC1s = np.array([0.401508, 0.376, 0.376, 0.69, 1, 0.52, 0.489, 0.401508])

times2 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
mGIs = np.array([0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789])

times3 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
mPRR3s = np.array([0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205])

spl1 = splrep(times1, mTOC1s, per=True)
spl2 = splrep(times2, mGIs, per=True)
spl3 = splrep(times3, mPRR3s, per=True)

def mTOC1(t):
    t_ = t % 24
    a = splev([t_], spl1)[0]
    return  np.maximum(a,0)#value cannot be zero

def dmTOC1dt(t):
    t_ = t % 24
    b = 0
    if mTOC1(t_) != 0:
        b = splev([t_], spl1, der = 1)[0]
    return b

def mGI(t):
    t_ = t % 24
    a = splev([t_], spl2)[0]
    return  np.maximum(a,0) #value cannot be zero

def dmGIdt(t):
    t_ = t % 24
    b = 0
    if mGI(t_) != 0:
        b = splev([t_], spl2, der = 1)[0]
    return b

def mPRR3(t):
    t_ = t % 24
    a = splev([t_], spl3)[0]
    return  np.maximum(a,0) #value cannot be zero

def dmPRR3dt(t):
    t_ = t % 24
    b = 0
    if mGI(t_) != 0:
        b = splev([t_], spl3, der = 1)[0]
    return b

x = np.linspace(0, 24, 2400)
y1 = mTOC1(x)
y2 = mGI(x)
y3 = mPRR3(x)
plt.plot(x, y1)
plt.plot(times1, mTOC1s, 'ro')
plt.suptitle('mTOC1s')
plt.show()

# def fOde(theta, x):
#     time = x[:, 0]
#     T = x[:, 1]
#     Zd = x[:, 2]
#     G = x[:, 3]
#     P = x[:, 4]
#     dXdt = np.zeros(shape=np.shape(x))
#     dXdt[:,0] = 1
#     dXdt[:,1] = mTOC1(time) - theta[0]*T
#     dXdt[:,2] = 1 - theta[1]*Zd
#     dXdt[:,3] = mGI(time) - theta[2]* G
#     dXdt[:,4] = mPRR3(time) - theta[3]* P
#     return dXdt

#
# def fOdeDx(theta, x):
#     resultDx = np.zeros(shape=[np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])
#     time = x[:, 0]
#     T = x[:, 1]
#     Zd = x[:, 2]
#     G = x[:, 3]
#     P = x[:, 4]
#     resultDx[:,0,1] = dmTOC1dt(time)
#     resultDx[:,1,1] = -theta[0]
#
#     resultDx[:,2,2] = - theta[1]
#
#     resultDx[:,0,3] = dmGIdt(time)
#     resultDx[:,3,3] = - theta[2]
#
#     resultDx[:,0,4] = dmPRR3dt(time)
#     resultDx[:,4,4] = - theta[3]
#
#     return resultDx
#
#
# def fOdeDtheta(theta, x):
#     resultDtheta = np.zeros(shape=[np.shape(x)[0], np.shape(theta)[0], np.shape(x)[1]])
#     time = x[:, 0]
#     T = x[:, 1]
#     Zd = x[:, 2]
#     G = x[:, 3]
#     P = x[:, 4]
#
#
#     resultDtheta[:,0,1] = -T
#     resultDtheta[:,1,2] = -Zd
#     resultDtheta[:,2,3] = -G
#     resultDtheta[:,3,4] = -P
#     return resultDtheta
#
#
# hes1_system = ode_system("Hes1-log-python", fOde, fOdeDx, fOdeDtheta,
#                          thetaLowerBound=np.ones(4) * 0, thetaUpperBound=np.ones(7) * 20)
