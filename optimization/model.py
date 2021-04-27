"""
System of ODEs for Plant Circadian Clocks

functions: odes, Euler_odes, simple_odes, Euler_simple_odes
"""
import numpy as np
from scipy.stats import levy_stable
import torch
import torch.nn as nn


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mTOC1(t):
    """
    Interpolated function for mRNA of TOC1
    Not necessarily backpropagatable!
    """
    times = np.array([0, 1, 5, 9, 13, 17, 21, 24])
    mTOC1s = np.array([0.401508, 0.376, 0.376, 0.69, 1, 0.52, 0.489, 0.401508])

    t_ = t % 24
    # print("TOC:", np.interp(t_, times, mTOC1s))
    return np.interp(t_, times, mTOC1s)


def mGI(t):
    """
    Interpolated function for mRNA of GI
    Not necessarily backpropagatable!
    """
    times = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    mGIs = np.array([0.0535789, 0.277942, 0.813305, 1., 0.373043, 0.00648925, 0.00439222, 0.0122333, 0.0535789])

    t_ = t % 24
    # print("GI:", np.interp(t_, times, mGIs))
    return np.interp(t_, times, mGIs)


def mPRR3(t):
    """
    Interpolated function for mRNA of PRR3
    Not necessarily backpropagatable!
    """
    times = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
    mPRR3s = np.array([0.010205, 0.00916596, 0.126271, 0.801952, 1., 0.091304, 0.0357569, 0.022007, 0.010205])

    t_ = t % 24
    # print("PRR3:", np.interp(t_, times, mPRR3s))
    return np.interp(t_, times, mPRR3s)


def light(t):
    t_ = t % 24
    return 1 if t_ <= 12 else 0


def odes(t, vars, params):
    """
    Full version of the ODEs!

    :param vars: 14 variables!!
    0  1    2    3  4  5     6     7   8   9     10    11    12    13
    T, Z_d, Z_l, G, P, TZ_d, TZ_l, TG, TP, Z_dG, Z_dP, Z_lG, Z_lP, GP

    :param params: 30 parameters!!
    0    1    2    3    4    5     6     7    8
    t_T, t_Z, t_G, t_P, d_T, d_Zd, d_Zl, d_G, d_P,
    9      10     11    12    13     14     15     16     17
    d_TZd, d_TZl, d_TG, d_TP, d_ZdG, d_ZdP, d_ZlG, d_ZlP, d_GP
    18   19   20
    k_f, k_l, k_d
    21     22     23    24    25     26     27     28     29
    k_TZd, k_TZl, k_TG, k_TP, k_ZdG, k_ZdP, k_ZlG, k_ZlP, k_GP

    :return: gradient flows at time t, in order of vars
    """
    assert len(vars) == 14
    assert len(params) == 30

    # gradient flows
    output = torch.zeros(14)

    output[0] = params[0] * mTOC1(t) - params[18] * (
                vars[0] * vars[1] + vars[0] * vars[2] + vars[0] * vars[3] + vars[0] * vars[4]) + \
                params[21] * vars[5] + params[22] * vars[6] + params[23] * vars[7] + params[24] * vars[8] - params[4] * \
                vars[0]

    output[1] = params[1] - params[18] * (vars[1] * vars[0] + vars[1] * vars[3] + vars[1] * vars[4]) + \
                params[9] * vars[5] + params[25] * vars[9] + params[26] * vars[10] - params[5] * vars[1] - \
                params[19] * light(t) * vars[1] + params[20] * (1 - light(t)) * vars[2]

    output[2] = params[19] * light(t) * vars[1] - params[20] * (1 - light(t)) * vars[2] - \
                params[18] * (vars[2] * vars[0] + vars[2] * vars[3] + vars[2] * vars[4]) + \
                params[10] * vars[6] + params[27] * vars[11] + params[16] * vars[12] - params[6] * vars[2]

    output[3] = params[2] * mGI(t) - params[18] * (
                vars[3] * vars[0] + vars[3] * vars[1] + vars[3] * vars[2] + vars[3] * vars[4]) + \
                params[23] * vars[7] + params[25] * vars[9] + params[27] * vars[11] + params[29] * vars[13] - params[
                    7] * vars[3]

    output[4] = params[3] * mPRR3(t) - params[18] * (
                vars[4] * vars[0] + vars[4] * vars[1] + vars[4] * vars[2] + vars[4] * vars[3]) + \
                params[24] * vars[8] + params[26] * vars[10] + params[28] * vars[12] + params[29] * vars[13] - params[
                    8] * vars[4]

    output[5] = params[18] * vars[0] * vars[1] - params[21] * vars[5] - params[9] * vars[5]

    output[6] = params[18] * vars[0] * vars[2] - params[22] * vars[6] - params[10] * vars[6]

    output[7] = params[18] * vars[0] * vars[3] - params[23] * vars[7] - params[11] * vars[7]

    output[8] = params[18] * vars[0] * vars[4] - params[24] * vars[8] - params[12] * vars[8]

    output[9] = params[18] * vars[1] * vars[3] - params[25] * vars[9] - params[13] * vars[9]

    output[10] = params[18] * vars[1] * vars[4] - params[26] * vars[10] - params[14] * vars[10]

    output[11] = params[18] * vars[2] * vars[3] - params[27] * vars[11] - params[15] * vars[11]

    output[12] = params[18] * vars[2] * vars[4] - params[28] * vars[12] - params[16] * vars[12]

    output[13] = params[18] * vars[3] * vars[4] - params[29] * vars[13] - params[17] * vars[13]

    return output


def Euler_odes(params, step_size=0.01):
    step_size = torch.tensor(step_size)

    # COMPUTE initial condition (torch tensor)
    # order of proteins: T, Z, G, P
    # with torch.no_grad():
    proteins_init = torch.zeros(14)
    t = torch.tensor(0.0)
    while t <= 24 * 7:
        # take the step
        proteins_init = proteins_init + step_size * odes(t, proteins_init, params)
        t += step_size

    # print("proteins_init: ", proteins_init)

    # time stamps for each protein
    TZ_stamp = [1, 5, 9, 13, 17, 21]
    GP_stamp = [0, 3, 6, 9, 12, 15, 18, 21, 24]

    # output
    T_output, Z_output = torch.zeros(6), torch.zeros(6)
    G_output, P_output = torch.zeros(9), torch.zeros(9)

    # Euler method loop
    # TODO: Other adaptive numerical methods?
    proteins = proteins_init.clone()
    t = torch.tensor(0.0)
    i, j = 0, 0
    while t <= 24:
        if t in TZ_stamp:
            T_output[i] = proteins[0]
            Z_output[i] = proteins[1]
            i += 1
        if t in GP_stamp:
            G_output[j] = proteins[3]
            P_output[j] = proteins[4]
            j += 1

        # take the step
        proteins = proteins + step_size * odes(t, proteins, params)
        t += step_size

    return T_output, Z_output, G_output, P_output


def loss(params, optimizer=None, iter=1, gradient=True):
    # Experimental datas
    TOC1_exp = torch.tensor([0.0649, 0.0346, 0.29, 0.987, 1, 0.645])
    ZTL_dark_exp = torch.tensor([0.115, 0.187, 0.445, 1., 0.718, 0.56])
    GI_exp = torch.tensor([0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325])
    PRR3_exp = torch.tensor([0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436])

    # if gradient:
    #     # Add noise (cf. SGLD, Neelakantan et al., 2016)
    #     lr = get_lr(optimizer)
    #     gamma = 0.55
    #     params.data = params.data + lr * np.sqrt(lr / ((1 + iter)*gamma)) * torch.randn(30)
        # params.data = params.data + np.sqrt(2 * get_lr(optimizer) / (1 + iter)) * torch.randn(30)
        # alpha, sigma = torch.rand(1)*(1 + 1/iter) + (1 - 1/iter), lr / ((1 + iter)*gamma)
        # params.data = params.data + lr * sigma * torch.tensor(levy_stable.rvs(alpha=alpha, beta=0, loc=0, size=30))
    # Output from ODEs
    TOC1_output, ZTL_dark_output, GI_output, PRR3_output = Euler_odes(params)

    # Define l2-loss
    l2_loss = nn.MSELoss()

    total_loss = l2_loss(TOC1_exp, TOC1_output) + l2_loss(ZTL_dark_exp, ZTL_dark_output) + l2_loss(GI_exp, GI_output) \
                 + l2_loss(PRR3_exp, PRR3_output)

    total_loss.backward()

    # Total loss
    return total_loss, params.grad.numpy()


def PSO_loss(params):
    output = []
    for param in params:
        params_ = torch.tensor(param)
        output.append(loss(params_, gradient=False)[0].item())
    return output


def basinhopping_loss(params):
    total_loss, total_grad = loss(torch.tensor(params, requires_grad=True), gradient=False)
    return total_loss.item(), total_grad
