import numpy as np
import sympy as sym
sym.init_printing(use_latex='matplotlib')
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
import pyro.distributions as dist
import torch.distributions as D
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from inference.inference import prepare_symbolic_plant, run_inference
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians
from scipy.interpolate import splrep, splev
from sympy import interpolating_spline


times1 = np.array([0, 1, 5, 9, 13, 17, 21, 24], dtype=float)
mTOC1s = np.array([0.401508, 0.376, 0.376, 0.69, 1, 0.52, 0.489, 0.401508])

times2 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24], dtype=float)
mGIs = np.array([0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789])

times3 = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24], dtype=float)
mPRR3s = np.array([0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205])


def mTOC1(t):
    return interpolating_spline(1, t, times1, mTOC1s)

def mGI(t):
    return interpolating_spline(1, t,times2, mGIs)

def mPRR3(t):
    return interpolating_spline(1, t,times3, mPRR3s)

### Define the LNA coupled ODE system ###
def r(y, t, p):
    Gtil, Ptil = y
    #    d_t, d_G, d_zd, d_P = p
    d_G, d_P = p
    #    dTtil_dt =  mTOC1(t) -d_t* Ttil
    dGtil_dt = mGI(t) - d_G * Gtil
    #    dZdtil_dt = 1 - d_zd * Zdtil
    dPtil_dt = mPRR3(t) - d_P * Ptil
    #    return dTtil_dt, dGtil_dt, dZdtil_dt, dPtil_dt
    return dGtil_dt, dPtil_dt

### Define generative model ### 
class PlantModel(PyroModule):
    def __init__(self, ode_op, ode_model):        
        super(PlantModel, self).__init__()
        self._ode_op = ode_op
        self._ode_model = ode_model
        # TODO: Incorporate appropriate priors (cf. MATALB codes from Daewook)
        self.ode_params1 = PyroSample(dist.Gamma(2,1))
        self.ode_params2 = PyroSample(dist.Gamma(2,1))
        # self.ode_params3 = PyroSample(dist.Beta(1,2))
        
    def forward(self, data): 
        p1 = self.ode_params1.view((-1,))
        p2 = self.ode_params2.view((-1,))
#        p3 = self.ode_params3.view((-1,))
        ode_params = torch.stack([p1, p2], dim=1)
        simple_sim = self._ode_op.apply(ode_params, (self._ode_model,))
        for i in range(1, len(data)):
            z_start = torch.stack([*data[i-1,:],*torch.zeros(0)])
            z_cov = self._ode_op.apply(torch.stack([p1,p2], dim=1), (self._ode_model, z_start))[...,-1,:]
            
            pyro.sample("obs_{}".format(i), dist.Exponential(simple_sim[..., i, 1]), obs=data[i])
            
        return z_cov 

def plot_marginals(vb_params, mc_params, param_names, real_params=None, rows=4):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize":(9,9),"font.size":16,"axes.titlesize":16,"axes.labelsize":16,
           "xtick.labelsize":15, "ytick.labelsize":15},style="white")

    for i, p in enumerate(param_names):        
        plt.subplot(rows, 2, i+1)
        if real_params is not None:
            plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i==0:
            sns.kdeplot(vb_params[:, i], color='magenta', linewidth = 2.5, label='Variational')
            sns.kdeplot(mc_params[:, i], color='orange', linewidth = 2.5, label='MCMC')
        else:
            sns.kdeplot(vb_params[:, i], linewidth = 2.5, color='magenta')
            sns.kdeplot(mc_params[:, i], linewidth = 2.5, color='orange')

            if i % 2 == 0:
                plt.ylabel('Frequency')
            plt.xlabel(param_names[i])
            if i < 1:
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2, fontsize=18)
        plt.subplots_adjust(hspace=0.7)
        plt.tight_layout()
        plt.show()
        plt.close()

    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
            description='Fit stochastic Lotka-Volterra model')
        parser.add_argument('--iterations', type=int, default=2000, metavar='N',
                            help='number of VI iterations')
        parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                            help='number of draws from variational posterior ')
        args = parser.parse_args()
        print('Iters: \n', args.iterations)
        print('q_samples: \n', args.num_qsamples)
        ### Generate the symbolic system and define the data generated earlier ###
        _rhs = r
        _y, _p = sym.symbols('y:2'), sym.symbols('p:2')
        rhs_f, jac_x_f, jac_p_f = prepare_symbolic_plant(_rhs, _y, _p, _t)
        times = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])

        data = np.array([
            [0.237939, 0.021049],  # 0
            [np.nan, 0.0711328],  # 3
            [0.365812, 0.128753],  # 6
            [0.913379, 0.574524],  # 9
            [1., 1.],  # 12
            [0.425148, 0.587505],  # 15
            [0.208709, 0.371859],  # 18
            [0.0937085, 0.355726],  # 21
            [0.096325, 0.104436],  # 24
        ])
        Gtil, Ptil = data[:, 0], data[:, 1]
        Y = data

        ### Run inference ###
        param_names = [r"$d_G$", r"$d_P$"]
        if not args.adjoint:

            print('Using VJP by Forward Sensitivity')
            plant_ode_model = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 2, 2, \
                                                       times, 1e-5, 1e-6, [0.237939, 0.021049])
            method = 'VI'
            lr = 0.5
            vb_samples = run_inference(Y, PlantModel, plant_ode_model, method, iterations=args.iterations, \
                                       lr=lr, num_particles=1, num_samples=args.num_qsamples, \
                                       return_sites=("ode_params1", "ode_params2"))
            vb_params_for = np.concatenate((vb_samples['ode_params1'][:, None].detach().numpy(),
                                            vb_samples['ode_params2'][:, None].detach().numpy()
                                            ), axis=1)
        else:
            print('Using VJP by Adjoint Sensitivity')
            plant_ode_model = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 2, 2,
                                                         times, 1e-5, 1e-6, [0.237939, 0.021049])

            vb_samples = run_inference(Y, LNAGenModel, lna_ode_model, method, iterations=args.iterations, \
                                       lr=lr, num_particles=1, num_samples=args.num_qsamples, \
                                       return_sites=("ode_params1", "ode_params2"))
            vb_params_adj = np.concatenate((vb_samples['ode_params1'][:, None].detach().numpy(),
                                            vb_samples['ode_params2'][:, None].detach().numpy()
                                            ), axis=1)

            plot_marginals(vb_params_for, vb_params_adj, param_names, rows=2)