"""

"""
import numpy as np
import torch
import torch.optim as optim
import model_simple
from datetime import datetime

# Set device (preferably GPU)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# verbose
verbose = True

# Initialize parameters randomly
# TODO: Is this init scheme valid...?
# tmp = abs(torch.randn(4))
tmp = torch.ones(4)
params = torch.tensor(tmp, requires_grad=True)
print("INITIAL params: ", params)

# Initialize optimizer
optimizer = optim.AdamW([params], lr=0.1)
optimizer_name = "AdamW"
# optimizer = optim.Adadelta([params])
# optimizer = optim.RMSprop([params], lr=0.1, momentum=0.1)
lbfgs = False

# optimizer_name = "LBFGS"
# optimizer = optim.LBFGS([params], lr=1)


def closure():
    optimizer.zero_grad()
    output = model_simple.loss(params)
    output.backward()
    return output


# lbfgs = True

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

iter = 1
eps = 1e-10  # threshold for stopping optimization
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

        output = model_simple.loss(params)

        curr_loss = output.item()
        if verbose and iter % 1 == 0:
            print("Iteration {:03d} | loss: {}".format(iter, curr_loss))

        output.backward()
        optimizer.step()
        scheduler.step()

    if curr_loss < eps or abs(prev_loss - curr_loss) < eps:
        break

    prev_loss = curr_loss
    iter += 1

print("FINAL params: ", params)
print("FINAL loss: ", curr_loss)
print("ITER required: ", iter)
np.savetxt("final_params_simple/{}_{}.csv".format(optimizer_name, datetime.now().strftime("%Y%m%d_%H%M%S")),
           params.detach().numpy())
