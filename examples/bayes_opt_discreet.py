# this is a simple example of Bayesian optimization using BoTorch and GPyTorch
# it uses the DropWave function as the objective function
# this is an example script for categorical optimization

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from botorch.models.kernels import CategoricalKernel
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
import torch
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from gpytorch.constraints import GreaterThan
from botorch.test_functions import DropWave
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys




def get_fitted_gp_model(X: torch.Tensor, Y: torch.Tensor, **tkwargs) -> SingleTaskGP:
    """
    Create and fit a GP model for a single objective.
    """
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
    )
    covar_module = ScaleKernel(
        base_kernel=MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]),
    )
    gp_model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=1),
        likelihood=likelihood,
    )
    mll = ExactMarginalLogLikelihood(model=gp_model, likelihood=gp_model.likelihood)
    fit_gpytorch_mll(mll)
    return gp_model


dw = DropWave()
def f(x):
    """
    Define the objective function to be minimized.
    """
    return -dw(x)


# Generate initial data (randomly sample from X)
X_initial = torch.tensor([[-3.0]], dtype=torch.float64)
Y_initial = torch.tensor([[f(x)] for x in X_initial], dtype=torch.float64)

print("X_initial:", X_initial)
print("Y_initial:", Y_initial)
print("X_initial shape:", X_initial.shape)
print("Y_initial shape:", Y_initial.shape)


frames = []

# Setup the figure once
fig, ax = plt.subplots(figsize=(8, 4))
X_plot = torch.linspace(-5.12, 5.12, 200).unsqueeze(1).double()
Y_plot = torch.tensor([[f(x)] for x in X_plot], dtype=torch.float64)
ax.plot(X_plot.numpy(), Y_plot.numpy(), label="True Function", alpha=0.4, color='gray')
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(Y_plot.min() - 0.1, Y_plot.max() + 0.1)
scat = ax.scatter([], [], color='blue', label="Observed Points")
ax.set_title("Bayesian Optimization Progress")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()

def update(frame):
    global X_initial, Y_initial
    candidates = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(1).double()
    model = get_fitted_gp_model(X_initial, Y_initial)
    best_value = Y_initial.max()
    EI = LogExpectedImprovement(model=model, best_f=best_value)
    new_point_analytic, _ = optimize_acqf_discrete(acq_function=EI, q=1, choices=candidates)

    X_initial = torch.cat([X_initial, new_point_analytic])
    Y_initial = torch.tensor([[f(x)] for x in X_initial], dtype=torch.float64)

    scat.set_offsets(torch.cat([X_initial, Y_initial], dim=1).numpy())
    ax.set_title(f"Step {frame+1} - Best value: {Y_initial.max().item():.4f}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=50, blit=False)

# Save the animation
ani.save("bo_animation.mp4", writer=animation.FFMpegWriter(fps=5))
'''
for i in range(50):
    canadates = torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(1).double()
    model = get_fitted_gp_model(X_initial, Y_initial)
    best_value = Y_initial.max()
    EI = LogExpectedImprovement(model=model, best_f=best_value)
    new_point_analytic, _ = optimize_acqf_discrete(
        acq_function=EI,
        q=1,
        choices=canadates
    )
    print(f"New point: {new_point_analytic.item()}")
    X_initial = torch.cat([X_initial, new_point_analytic])
    Y_initial = f(X_initial).unsqueeze(-1)

    # Update the plot
    scatter.remove()
    scatter = ax.scatter(X_initial.numpy(), Y_initial.numpy(), color='blue')
    ax.set_title(f"Step {i+1} - Best value: {Y_initial.max().item():.4f}")
    plt.pause(0.1)  # brief pause to allow the plot to update


print(f'Best value achieved after {len(X_initial) - 3} BO steps: {Y_initial.max().item()}')

plt.ioff()
plt.show()
'''