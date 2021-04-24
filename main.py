"""

"""
import numpy as np
import torch
import torch.optim as optim
from pyswarms.single.global_best import GlobalBestPSO
from scipy.optimize import basinhopping
from scipy.stats import levy_stable
import model
from datetime import datetime

# Set device (preferably GPU)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# verbose
verbose = True

# gradient based: torch.optim
# not gradient based: PSO (pyswarms), basinhopping(scipy.optimize)
gradient_based = True
optimizer_name = "basinhopping"

# lr scheduling
schedule = False

# Initialize parameters randomly
# TODO: Is this init scheme valid...?
tmp = abs(torch.randn(30))*10
params = torch.tensor(tmp, requires_grad=True)
# params = torch.ones(30, requires_grad=True)
print("INITIAL params: ", params)


if gradient_based:
    # Initialize optimizer
    optimizer = optim.SGD([params], lr=0.1, momentum=0.9, nesterov=True)
    optimizer_name = "SGD"
    # optimizer = optim.Adam([params], lr=0.5, amsgrad=True)
    # optimizer_name = "Adam"
    # optimizer = optim.Adadelta([params])
    # optimizer = optim.RMSprop([params], lr=0.1, momentum=0.1)
    lbfgs = False

    init = np.concatenate((10 * np.ones(18), np.array([100, 10, 10]), 100 * np.ones(9)))
    params_ = torch.tensor(init, requires_grad=True)
    output = model.loss(params_, optimizer, iter)
    print(params_)
    print(output.item())
    # for _ in range(20):
    #     tmp = abs(torch.tensor(levy_stable.rvs(alpha=1, beta=0, loc=0, size=30)))
    #     params_ = torch.tensor(tmp, requires_grad=True)
    #     output = model.loss(params_, optimizer, iter)
    #     print(params_)
    #     print(output.item())
    raise ValueError("여기까")

    # optimizer_name = "LBFGS"
    # optimizer = optim.LBFGS([params], lr=1)
    # lbfgs = True

    def closure():
        optimizer.zero_grad()
        output, _ = model.loss(params)
        output.backward()
        return output


    if schedule:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 25, 50, 100], gamma=0.2)

    iter = 1
    eps = 1e-1  # hard threshold for stopping optimization
    eps_ = 1e-10  # convergence threshold for stopping optimization
    prev_loss, curr_loss = 0, 0
    while True:
        # print("grad: {}".format(params.grad))
        if lbfgs:
            curr_loss = closure().item()
            if verbose and iter % 1 == 0:
                print("Iteration {:03d} | loss: {}".format(iter, curr_loss))
            optimizer.step(closure)
        else:
            optimizer.zero_grad()

            output, _ = model.loss(params, optimizer, iter)

            curr_loss = output.item()
            if verbose and iter % 1 == 0:
                print("Iteration {:03d} | loss: {}".format(iter, curr_loss))

            # .backward() done in mode.loss(params)!
            # output.backward()
            optimizer.step()
            if schedule:
                scheduler.step()

        if curr_loss < eps:
        # if curr_loss < eps or abs(prev_loss - curr_loss) < eps_:
            break

        prev_loss = curr_loss
        iter += 1

    print("FINAL params: ", params)
    print("FINAL loss: ", curr_loss)
    print("ITER required: ", iter)
    np.savetxt("final_params/{}_{}.csv".format(optimizer_name, datetime.now().strftime("%Y%m%d_%H%M%S")),
               params.detach().numpy())
else:
    if optimizer_name == "PSO":
        # Hyperparameters for PSO
        # c1: cognitive parameter, c2: social parameter, w: inertia parameter
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # Initialize optimizer
        optimizer = GlobalBestPSO(n_particles=20, dimensions=30, options=options)

        # Perform optimization
        cost, pos = optimizer.optimize(model.PSO_loss, iters=1000)

        print("FINAL params: ", pos)
        print("FINAL loss: ", cost)
        print("ITER required: ", iter)
        np.savetxt("final_params/{}_{}.csv".format(optimizer_name, datetime.now().strftime("%Y%m%d_%H%M%S")),
                   np.array(pos))

    elif optimizer_name == "basinhopping":
        param0 = np.ones(30)
        minimizer_kwargs = {"method": "Newton-CG", "jac": True}

        def print_log(params, f, accepted):
            print("at minimum %.8f accepted %s" % (f, int(accepted)))

        ret = basinhopping(model.basinhopping_loss, param0, minimizer_kwargs=minimizer_kwargs, niter=5, callback=print_log)
        print("global minimum: params = {}, loss(params) = {}".format(ret.x, ret.fun))
        np.savetxt("final_params/{}_{}.csv".format(optimizer_name, datetime.now().strftime("%Y%m%d_%H%M%S")), ret.x)
