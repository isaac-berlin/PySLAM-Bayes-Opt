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
import _goal
import datetime
import pandas as pd


def get_fitted_gp_model(X: torch.Tensor, Y: torch.Tensor, **tkwargs) -> SingleTaskGP:
    """
    Create and fit a GP model for a single objective.
    """
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
        noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
    )
    covar_module = ScaleKernel(
        base_kernel=CategoricalKernel(ard_num_dims=X.shape[-1]),
        #base_kernel=MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]),
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


def f(x):
    x = x.item()
    x = int(x)
    
    total = 0
    trials = 1
    for i in range(trials):
        _goal.set_options(x)
        res = None
        trial = 0
        while res is None:
            trial += 1
            print(f"[{datetime.datetime.now().strftime('%I:%M:%S %p')}]Trial {trial} START with {x} ORB features")
            _goal.run_slam()
            res = _goal.check_results()
        if res is not None: # rerun until SLAM works properly
            print(f"[{datetime.datetime.now().strftime('%I:%M:%S %p')}]Trial {trial} END with {x} ORB features, result: {res}")
            total += res
    return (total/trials) * -1



priors_path = r"/home/isaac/Desktop/prior.csv" # required to run the script (run a few times manually)
priors = pd.read_csv(priors_path)
X_initial = torch.tensor(priors["feats"].values, dtype=torch.float64).unsqueeze(1)
Y_initial = torch.tensor(priors["RMSE"].values, dtype=torch.float64).unsqueeze(1)

candidates = torch.arange(900, 3001, 100, dtype=torch.float64).unsqueeze(1)

for i in range(30):

    model = get_fitted_gp_model(X_initial, Y_initial)
    best_value = Y_initial.max()
    EI = LogExpectedImprovement(model=model, best_f=best_value)
    new_point_analytic, _ = optimize_acqf_discrete(
        acq_function=EI,
        q=1,
        choices=candidates
    )
    new_point_analytic = new_point_analytic.item()
    new_point_analytic = int(new_point_analytic)
    new_point_analytic = torch.tensor([[new_point_analytic]], dtype=torch.float64)
    print(f"New point: {new_point_analytic.item()}")
    X_initial = torch.cat([X_initial, new_point_analytic])
    next_y = f(new_point_analytic)
    Y_initial = torch.cat([Y_initial, torch.tensor([[next_y]], dtype=torch.float64)])
    save_path = "bayes_optim_results.pt"
    torch.save({
        'X_initial': X_initial,
        'Y_initial': Y_initial,
    }, save_path)
    print(f"Results saved to {save_path}") # save the results to a pt tensor