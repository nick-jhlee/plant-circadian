import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import seaborn as sns

sym.init_printing(use_latex='matplotlib')
import time as timer
import torch
import pyro
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import AdagradRMSProp
# from pyro.infer.autoguide.initialization import init_to_median
from pyro.infer.autoguide import init_to_median, init_to_sample
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from torch_grad.torch_custom_op import ForwardSensOp, AdjointSensOp
from ode_systems.forward_sensitivity_solvers import ForwardSensManualJacobians, ForwardSensTorchJacobians
from ode_systems.adjoint_sensitivity_solvers import AdjointSensManualJacobians, AdjointSensTorchJacobians


def prepare_symbolic(rhs, y, p):
    ydot = rhs(y, None, p)
    t = sym.symbols('t')
    rhs_f = sym.lambdify((y, t, p), ydot)
    jac_x = sym.Matrix(ydot).jacobian(y)
    jac_p = sym.Matrix(ydot).jacobian(p)
    jac_x_f = sym.lambdify((y, p), jac_x)
    jac_p_f = sym.lambdify((y, p), jac_p)
    print('\n\n The velocity field df/dt : \n\n', ydot)
    print('\n\n Jacobian of the velocity wrt X(t), df/d X: \n\n', jac_x)
    print('\n\n Jacobian of the velocity wrt theta, df/d theta: \n\n', jac_p)
    return rhs_f, jac_x_f, jac_p_f


def prepare_symbolic_plant(rhs, y, p, t=None):
    ydot = rhs(y, t, p)
    # t = sym.symbols('t')
    rhs_f = sym.lambdify((y, t, p), ydot)
    jac_x = sym.Matrix(ydot).jacobian(y)
    jac_p = sym.Matrix(ydot).jacobian(p)
    jac_x_f = sym.lambdify((y, t, p), jac_x)
    jac_p_f = sym.lambdify((y, t, p), jac_p)
    print('\n\n The velocity field df/dt : \n\n', ydot)
    print('\n\n Jacobian of the velocity wrt X(t), df/d X: \n\n', jac_x)
    print('\n\n Jacobian of the velocity wrt theta, df/d theta: \n\n', jac_p)
    return rhs_f, jac_x_f, jac_p_f


def run_inference(data, gen_model, ode_model, method, iterations=10000, num_particles=1, num_samples=1000, warmup_steps=500, init_scale=0.1,
                  seed=12, lr=0.5, return_sites="_RETURN", directory=None, param_freq=500, plot_freq=1000, param_names=None):
    torch_data = torch.tensor(data, dtype=torch.float)
    if isinstance(ode_model, ForwardSensManualJacobians) or \
            isinstance(ode_model, ForwardSensTorchJacobians):
        ode_op = ForwardSensOp
    elif isinstance(ode_model, AdjointSensManualJacobians) or \
            isinstance(ode_model, AdjointSensTorchJacobians):
        ode_op = AdjointSensOp
    else:
        raise ValueError('Unknown sensitivity solver: Use "Forward" or "Adjoint"')
    model = gen_model(ode_op, ode_model)
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    if method == 'VI':

         # guide = AutoDiagonalNormal(model, init_scale=init_scale)
        guide = AutoMultivariateNormal(model, init_scale=init_scale)
        optim = AdagradRMSProp({"eta": lr})
        if num_particles == 1:
            svi = SVI(model, guide, optim, loss=Trace_ELBO())
        else:
            svi = SVI(model, guide, optim, loss=Trace_ELBO(num_particles=num_particles,
                                                           vectorize_particles=True))
        loss_trace = []
        t0 = timer.time()
        for j in range(iterations):
            loss = svi.step(torch_data)
            loss_trace.append(loss)

            if j % 500 == 0 or j == iterations-1:
                print("[iteration %04d] loss: %.4f" % (j + 1, np.mean(loss_trace[max(0, j - 1000):j + 1])))

            # save params
            if j % param_freq == 0 or j == iterations-1:
                predictive = Predictive(model, guide=guide, num_samples=num_samples,
                                        return_sites=return_sites)  # "ode_params", "scale",
                vb_samples = predictive(torch_data)
                vb_params = np.concatenate((vb_samples['ode_params1'][:, None].detach().numpy(),
                                            vb_samples['ode_params2'][:, None].detach().numpy(),
                                            vb_samples['ode_params3'][:, None].detach().numpy(),
                                            vb_samples['ode_params4'][:, None].detach().numpy(),
                                            vb_samples['ode_params5'][:, None].detach().numpy(),
                                            vb_samples['ode_params6'][:, None].detach().numpy(),
                                            vb_samples['ode_params7'][:, None].detach().numpy(),
                                            vb_samples['ode_params8'][:, None].detach().numpy(),
                                            vb_samples['ode_params9'][:, None].detach().numpy(),
                                            vb_samples['ode_params10'][:, None].detach().numpy(),
                                            vb_samples['ode_params11'][:, None].detach().numpy(),
                                            vb_samples['ode_params12'][:, None].detach().numpy()
                                            ), axis=1)
                np.savetxt('{}/vb_params (iter {}).csv'.format(directory, j+1), vb_params, delimiter=',')
                np.savetxt('{}/vb_params_mean (iter {}).csv'.format(directory, j+1), np.mean(vb_params, axis=0), delimiter=',')
                # save plots
                if j % plot_freq == 0 or j == iterations-1:
                    plot_marginals_vb(vb_params, param_names, directory, j+1, rows=12)
        t1 = timer.time()
        print('VI time (including param sampling and saving): ', t1 - t0)
        # predictive = Predictive(model, guide=guide, num_samples=num_samples,
        #                         return_sites=return_sites)  # "ode_params", "scale",
        # vb_samples = predictive(torch_data)
        # return vb_samples
        return None

    elif method == 'NUTS':

        nuts_kernel = NUTS(model, adapt_step_size=True, init_strategy=init_to_median)

        # mcmc = MCMC(nuts_kernel, num_samples=iterations, warmup_steps=warmup_steps, num_chains=2)
        mcmc = MCMC(nuts_kernel, num_samples=iterations, warmup_steps=warmup_steps, num_chains=1)
        t0 = timer.time()
        mcmc.run(torch_data)
        t1 = timer.time()
        print('NUTS time: ', t1 - t0)
        hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
        return hmc_samples
    else:
        raise ValueError('Unknown method: Use "NUTS" or "VI"')


def plot_marginals_vb(vb_params, param_names, directory, iter, real_params=None, rows=4):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize": (9, 9), "font.size": 16, "axes.titlesize": 16, "axes.labelsize": 16,
                "xtick.labelsize": 15, "ytick.labelsize": 15}, style="white")

    # print(vb_params)
    for i, p in enumerate(param_names):
        plt.subplot(rows, 2, i + 1)
        if real_params is not None:
            plt.axvline(real_params[i], linewidth=2.5, color='black')
        if i == 0:
            sns.kdeplot(vb_params[:, i], color='magenta', linewidth=2.5, label='Variational')
        else:
            sns.kdeplot(vb_params[:, i], linewidth=2.5, color='magenta')

        if i % 2 == 0:
            plt.ylabel('Frequency')
        plt.xlabel(param_names[i])
        if i < 1:
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower center', ncol=2, fontsize=18)
    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.savefig('{}/param_posterior (iter {})'.format(directory, iter))
    # plt.show()
    # plt.close()