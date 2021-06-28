import numpy as np
import sympy as sym

sym.init_printing(use_latex='matplotlib')
import torch
import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule
import pyro.distributions as dist
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from inference.inference import prepare_symbolic_plant, run_inference
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians
from sympy import interpolating_spline

init_days = 1


def extend_time(times_init):
    times_init = np.array(times_init, dtype=float)
    times = np.array(times_init, dtype=float)
    if init_days < 1:
        return times
    for day in range(1, init_days + 2):
        times = np.append(times, times_init + 24 * day)
    return times


times1 = extend_time([0, 1, 5, 9, 13, 17, 21])
mTOC1s = np.array([0.401508, 0.376, 0.376, 0.69, 1, 0.52, 0.489] * (init_days + 2))

times2 = extend_time([0, 3, 6, 9, 12, 15, 18, 21])
mGIs = np.array([0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333] * (init_days + 2))

times3 = extend_time([0, 3, 6, 9, 12, 15, 18, 21])
mPRR3s = np.array([0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007] * (init_days + 2))


def mTOC1(t):
    return interpolating_spline(1, t, times1, mTOC1s)


def mGI(t):
    return interpolating_spline(1, t, times2, mGIs)


def mPRR3(t):
    return interpolating_spline(1, t, times3, mPRR3s)


### Define ODE right hand side ###
def r(y, t, p):
    #    Ttil, Gtil, Zdtil, Ptil = y
    Ttil, Ztil = y
    #    d_t, d_G, d_zd, d_P = p
    d_T, d_Z = p
    #    dTtil_dt =  mTOC1(t) -d_t* Ttil
    dTtil_dt = mTOC1(t) - d_T * Ttil
    #    dZdtil_dt = 1 - d_zd * Zdtil
    dZtil_dt = 1 - d_Z * Ztil
    #    return dTtil_dt, dGtil_dt, dZdtil_dt, dPtil_dt
    return dTtil_dt, dZtil_dt


### Define generative model ###
class PlantModel(PyroModule):
    def __init__(self, ode_op, ode_model):
        super(PlantModel, self).__init__()
        self._ode_op = ode_op
        self._ode_model = ode_model
        # TODO: Incorporate appropriate priors (cf. MATALB codes from Daewook)
        self.ode_params1 = PyroSample(dist.Gamma(1, 2))  # dT
        self.ode_params2 = PyroSample(dist.Gamma(1, 2))  # dZ
   #     self.ode_params3 = PyroSample(dist.Beta(0.5, 0.5))  # T0
   #     self.ode_params4 = PyroSample(dist.Beta(0.5, 0.5))  # Z0

    #        self.ode_params3 = PyroSample(dist.Gamma(2,1))
    #        self.ode_params4 = PyroSample(dist.Gamma(2,1))

    def forward(self, data):
        scale = pyro.sample("scale", dist.HalfNormal(0.001))
        sd = scale.view((-1,)).unsqueeze(1)
        # print("sd: ", sd)
        p1 = self.ode_params1.view((-1,))
        p2 = self.ode_params2.view((-1,))
        ode_params = torch.stack([p1, p2], dim=1)
        simple_sim = self._ode_op.apply(ode_params, (self._ode_model,))

        for i in range(len(data)):
            try:
                # TODO: Which distribution to use?
                # pyro.sample("obs_{}".format(i), dist.Exponential(simple_sim[..., i, 0]), obs=data[i])
                pyro.sample("obs_{}".format(i), dist.Normal(loc=simple_sim[..., i, :], scale=sd).to_event(1),
                            obs=data[i, :])
            except:
                print(simple_sim[..., i, :])
                print("ERROR (invalid parameter for Normal...!): ")
        return simple_sim


def plot_marginals(vb_params, mc_params, param_names, real_params=None, rows=4):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize": (9, 9), "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
                "xtick.labelsize": 15, "ytick.labelsize": 15}, style="white")

    # print(vb_params)
    np.savetxt('vb_params.csv', vb_params, delimiter=',')
    for i, p in enumerate(param_names):
        plt.subplot(rows, 2, i + 1)
        if real_params is not None:
            plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i == 0:
            sns.kdeplot(vb_params[:, i], color='magenta', linewidth=2.5, label='Variational')
            sns.kdeplot(mc_params[:, i], color='orange', linewidth=2.5, label='NUTS')
        else:
            sns.kdeplot(vb_params[:, i], linewidth=2.5, color='magenta')
            sns.kdeplot(mc_params[:, i], linewidth=2.5, color='orange')

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
        description='Fit Protein Transduction model')
    parser.add_argument('--adjoint', type=bool, default=False, metavar='N',
                        help='Method to compute VJP')
    parser.add_argument('--iterations', type=int, default=10000, metavar='N',
                        help='number of VI iterations')
    parser.add_argument('--num_qsamples', type=int, default=1000, metavar='N',
                        help='number of draws from variational posterior ')
    parser.add_argument('--num_samples', type=int, default=1000, metavar='N',
                        help='number of NUTS post warm-up samples')
    parser.add_argument('--warmup_steps', type=int, default=500, metavar='N',
                        help='number of NUTS warmup_steps')
    # parser.add_argument('--init_days', type=int, default=1, metavar='N',
    #                     help='number of days to be pre-computed for convergence to periodic function')
    args = parser.parse_args()

    ### Generate the symbolic system ###z
    _rhs = r
    _y, _p = sym.symbols('y:2'), sym.symbols('p:2')
    # TODO : input _t
    _t = sym.symbols('t')
    rhs_f, jac_x_f, jac_p_f = prepare_symbolic_plant(_rhs, _y, _p, _t)

    ### Input experimental data ###
    times = np.array([1, 5, 9, 13, 17, 21]) + (init_days * 24)
    data = np.array([
        [0.0649, 0.115],  # 1 + init_days * 24
        [0.0346, 0.187],  # 5 + init_days * 24
        [0.29, 0.445],  # 9 + init_days * 24
        [0.987, 1.],  # 13 + init_days * 24
        [1., 0.718],  # 17 + init_days * 24
        [0.645, 0.56],  # 21 + init_days * 24
    ])
    Ttil, Ztil = data[:, 0], data[:, 1]
    Y = data

    ### Run inference ###
    param_names = [r"$d_T$", r"$d_Z$"]
    if not args.adjoint:
        print('Using VJP by Forward Sensitivity')
        plant_ode_model = ForwardSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 2, 2,
                                                     times, 1e-5, 1e-6, [0.0649, 0.115])

        # plant_ode_model.set_unknown_y0()
        # method = 'NUTS'
        # NUTS_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
        #     iterations = args.num_samples, warmup_steps = args.warmup_steps)
        # mc_params=np.concatenate((NUTS_samples['ode_params1'][:,None],
        #                     NUTS_samples['ode_params2'][:,None],
        #                     NUTS_samples['ode_params3'][:,None]
        #                     ),axis=1)

        method = 'VI'
        lr = 0.5
        num_particles = 1
        vb_samples = run_inference(Y, PlantModel, plant_ode_model, method,
                                   iterations=args.iterations, num_samples=args.num_qsamples,
                                   lr=lr, num_particles=num_particles, return_sites=("ode_params1", "ode_params2"))
        vb_params = np.concatenate((vb_samples['ode_params1'][:, None].detach().numpy(),
                                    vb_samples['ode_params2'][:, None].detach().numpy()
                                    ), axis=1)

        # plot_marginals(vb_params, mc_params, param_names, rows=2)
        plot_marginals(vb_params, vb_params, param_names, rows=2)
    else:
        print('Using VJP by Adjoint Sensitivity')
        plant_ode_model = AdjointSensManualJacobians(rhs_f, jac_x_f, jac_p_f, 2, 2,
                                                     times, 1e-5, 1e-6, [0.0649, 0.115])

        # plant_ode_model.set_unknown_y0()
        # method = 'NUTS'
        # NUTS_samples = run_inference(Y, SIRGenModel, sir_ode_model, method, \
        #     iterations = args.num_samples, warmup_steps = args.warmup_steps)
        # mc_params=np.concatenate((NUTS_samples['ode_params1'][:,None],
        #                     NUTS_samples['ode_params2'][:,None],
        #                     NUTS_samples['ode_params3'][:,None]
        #                     ),axis=1)

        method = 'VI'
        lr = 0.5
        num_particles = 1
        vb_samples = run_inference(Y, PlantModel, plant_ode_model, method,
                                   iterations=args.iterations, num_samples=args.num_qsamples,
                                   lr=lr, num_particles=num_particles, return_sites=("ode_params1", "ode_params2"))
        vb_params = np.concatenate((vb_samples['ode_params1'][:, None].detach().numpy(),
                                    vb_samples['ode_params2'][:, None].detach().numpy()
                                    ), axis=1)
        # plot_marginals(vb_params, mc_params, param_names, rows=2)
        plot_marginals(vb_params, vb_params, param_names, rows=2)