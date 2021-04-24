"""
System of ODEs for Plant Circadian Clocks

functions: odes, Euler_odes, simple_odes, Euler_simple_odes
"""
import numpy as np
import torch
import torch.nn as nn


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


def simple_odes(t, vars, params):
    """
    Simple version of the ODEs!
    (dimers not accounted for + nondimensionalization)

    :param vars: only 4 variables!! [TOC1, ZTL_dark, GI, PRR3]
    :param params: only 4 parameters!! (translation rates for each protein)
    :return: gradient flows at time t [d[TOC1]/dt, d[ZTL_dark]/dt, d[GI]/dt, d[PRR3]/dt]
    """
    assert len(vars) == 14
    assert len(params) == 14

    # output = torch.zeros(4)
    #
    # output[0] = mTOC1(t) - params[0] * vars[0]
    # output[1] = 1 - params[1] * vars[1]
    # output[2] = mGI(t) - params[2] * vars[2]
    # output[3] = mPRR3(t) - params[3] * vars[3]
    # return output

    # gradient flows
    return torch.tensor([mTOC1(t), 1, mGI(t), mPRR3(t)]) - (params * vars)


def Euler_simple_odes(params, step_size=0.01):
    step_size = torch.tensor(step_size)

    # COMPUTE initial condition (torch tensor)
    # order of proteins: T, Z, G, P
    # with torch.no_grad():
    proteins_init = torch.zeros(4)
    t = torch.tensor(0.0)
    while t <= 24 * 10:
        # take the step
        proteins_init = proteins_init + step_size * simple_odes(t, proteins_init, params)
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
            G_output[j] = proteins[2]
            P_output[j] = proteins[3]
            j += 1

        # take the step
        proteins = proteins + step_size * simple_odes(t, proteins, params)
        t += step_size

    return T_output, Z_output, G_output, P_output


def loss(params):
    # Experimental datas
    TOC1_exp = torch.tensor([0.0649, 0.0346, 0.29, 0.987, 1, 0.645])
    ZTL_dark_exp = torch.tensor([0.115, 0.187, 0.445, 1., 0.718, 0.56])
    GI_exp = torch.tensor([0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325])
    PRR3_exp = torch.tensor([0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436])

    # Output from ODEs
    TOC1_output, ZTL_dark_output, GI_output, PRR3_output = Euler_simple_odes(params)

    # Define l2-loss
    l2_loss = nn.MSELoss()

    # Total loss
    return l2_loss(TOC1_exp, TOC1_output) + l2_loss(ZTL_dark_exp, ZTL_dark_output) + l2_loss(GI_exp, GI_output) \
           + l2_loss(PRR3_exp, PRR3_output)


def stochastic_loss(params, batch_size=15):
    # Experimental datas
    TOC1_exp = torch.tensor([0.0649, 0.0346, 0.29, 0.987, 1, 0.645])
    ZTL_dark_exp = torch.tensor([0.115, 0.187, 0.445, 1., 0.718, 0.56])
    GI_exp = torch.tensor([0.237939, 0.0842713, 0.365812, 0.913379, 1., 0.425148, 0.208709, 0.0937085, 0.096325])
    PRR3_exp = torch.tensor([0.021049, 0.0711328, 0.128753, 0.574524, 1., 0.587505, 0.371859, 0.355726, 0.104436])

    # Output from ODEs
    TOC1_output, ZTL_dark_output, GI_output, PRR3_output = Euler_simple_odes(params)

    # Define l2-loss
    l2_loss = nn.MSELoss()

    # Total loss
    return l2_loss(TOC1_exp, TOC1_output) + l2_loss(ZTL_dark_exp, ZTL_dark_output) + l2_loss(GI_exp, GI_output) \
           + l2_loss(PRR3_exp, PRR3_output)
