import torch
import gpytorch
import numpy as np
import pandas as pd
from gpytorch.constraints import Positive
from copy import deepcopy

def set_gpytorch_settings(dtype=torch.float64, use_cuda=True):
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(False)
    gpytorch.settings.fast_computations.log_prob._set_state(False)
    gpytorch.settings.fast_computations.solves._set_state(False)
    gpytorch.settings.cholesky_max_tries._set_value(100)
    gpytorch.settings.debug._set_state(False)
    gpytorch.settings.min_fixed_noise._set_value(
        float_value=1e-7, double_value=1e-7, half_value=1e-7
    )

    # Set default dtype
    torch.set_default_dtype(dtype)

    # Handle CUDA settings
    if use_cuda and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if dtype == torch.float64:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        elif dtype == torch.float32:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Default to float32
    else:
        if use_cuda and not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA is not available. Falling back to CPU.")
        if 'LAPACK' not in torch.__config__.show():
            raise RuntimeError(
                "PyTorch was not installed with LAPACK support. Please reinstall PyTorch with LAPACK support."
            )
        if dtype == torch.float64:
            torch.set_default_tensor_type(torch.DoubleTensor)
        elif dtype == torch.float32:
            torch.set_default_tensor_type(torch.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)  # Default to float32

    # For approximate likelihoods, like t or binomial
    gpytorch.settings.ciq_samples._set_state(False)
    gpytorch.settings.skip_logdet_forward._set_state(False)
    gpytorch.settings.num_trace_samples._set_value(0)
    gpytorch.settings.num_gauss_hermite_locs._set_value(300)
    gpytorch.settings.num_likelihood_samples._set_value(300)
    gpytorch.settings.deterministic_probes._set_state(True)


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def to_torch(array, device=None, dtype=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(array, dtype=dtype).to(device)


class MinMaxScaler:
    def __init__(self, mins=None, maxs=None):
        self.reset()
        self.mins = mins
        self.maxs = maxs

    def reset(self):
        self.mins = None
        self.maxs = None

    def fit(self, X):
        self.reset()
        if isinstance(X, pd.DataFrame):
            X = to_torch(X.values)
        elif isinstance(X, np.ndarray):
            X = to_torch(X)
        elif isinstance(X, torch.Tensor):
            X = X.to(dtype=torch.get_default_dtype())

        self.mins = torch.min(X, dim=0).values
        self.maxs = torch.max(X, dim=0).values

        return self.scale(X)

    def scale(self, X):
        if isinstance(X, np.ndarray):
            X = to_torch(X)
        elif isinstance(X, pd.DataFrame):
            X = to_torch(X.values)
        X_scaled = (X - self.mins) / (self.maxs - self.mins)
        return X_scaled

    def unscale(self, X_scaled):
        if isinstance(X_scaled, np.ndarray):
            X_scaled = to_torch(X_scaled)

        X = X_scaled.unsqueeze(-1) * (self.maxs - self.mins) + self.mins
        return X.squeeze(-1)


class NoScale:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def scale(self, X):
        return X

    def unscale(self, X_scaled):
        return X_scaled


class GPRModel(gpytorch.models.ExactGP):
    """
    Gaussian Process Regression (GPR) Model class.

    This class encapsulates the GPR model, providing a simple and intuitive API for fitting the model to data and making predictions. It interacts with the GPyTorch library to perform these operations.

    Attributes:
        - train_x (Tensor): Training data inputs.
        - train_y (Tensor): Training data outputs.
        - kernel (Kernel): The kernel function to use for the GP Model.
        - mean (Mean): The mean function to use for the GP model
        - scale_x (True/False): Whether to scale x variables to unit interval (helps with hyperparameter optimization).
    """

    def __init__(self, train_x=None, train_y=None, kernel=None, mean='constant', scale_x=True):

        self.scale_x = scale_x

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Positive())

        if scale_x is True:
            # Initialize and fit the MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            self.scaler = NoScale()

        # Case of no training data, e.g. for prior GPs
        if train_x is None and train_y is None:
            self.train_x_scaled = None

        # Scale and handle data
        else:
            if isinstance(train_x, np.ndarray):
                train_x = to_torch(train_x)
            if isinstance(train_y, np.ndarray):
                train_y = to_torch(train_y)
            if isinstance(train_x, torch.Tensor):
                train_x = train_x.to(dtype=torch.get_default_dtype())
            if isinstance(train_y, torch.Tensor):
                train_y = train_y.to(dtype=torch.get_default_dtype())
            self.scaler.fit(train_x)
            self.train_x_scaled = self.scaler.scale(train_x)

        self.train_x = train_x
        self.train_y = train_y

        # Correctly initialize according to gpytorch.models.ExactGP
        super(GPRModel, self).__init__(self.train_x_scaled, self.train_y, likelihood)

        # Initialize mean module
        if mean == 'constant':
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean is None:
            self.mean_module = gpytorch.means.ZeroMean()
        elif isinstance(mean, gpytorch.means.Mean):
            self.mean_module = deepcopy(mean)
        else:
            raise TypeError("The 'mean' parameter must be None or an instance of gpytorch.means.Mean.")

        self.kernel = deepcopy(kernel)

        self.predictions = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit_model(self, training_iterations=50, verbose=True, lr=0.1):
        """
        Fit the GPR model to the training data.

        Args:
            - train_x (Tensor): The training data inputs.
            - train_y (Tensor): The training data outputs.
            - training_iterations (int): The number of iterations for training the model.

        Returns:
            - self: The fitted GPR model.
        """

        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        # TODO: convergence tolerance
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self(self.train_x_scaled)

            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

        self.compute_bic()

        if verbose is True:
            print("Fitting complete.")
            print(f"--- ")
            print(f"--- final mll: {-loss:.4f}")
            print(f"--- num_params: {self.num_param}")
            print(f"--- BIC: {self.bic:.4f}")

        return self

    def simulate(self, x_sim, method='prior', type='f', return_type='numpy', n_paths=1):
        """
        This method generates samples from a multivariate normal distribution, either from the prior or the posterior.

        Parameters:
        x_sim (torch.Tensor): The x values for which the samples will be generated.
        method (str): Specifies whether to generate samples from the 'prior' or the 'posterior'. Default is 'prior'.
        type (str): Specifies the type of samples to generate - 'f' for the underlying GP (function values) and 'y' for predictions (observations). Default is 'f'.

        Returns:
        torch.Tensor: Generated samples.
        """
        if n_paths > 1:
            # TODO: address this case
            raise NotImplementedError("Currently supports simulating one sample path. You can rerun 'simulate' to get additional paths.")

        # Strangely, GPyTorch doesn't allow "train mode" (i.e. prior) if there is no training data.
        if self.train_x is None and self.train_y is None:
            self.eval()
        else:
            if method == 'prior':
                self.train()
            elif method == 'posterior':
                self.eval()

        if isinstance(x_sim, np.ndarray):
            x_sim = to_torch(x_sim)
        elif isinstance(x_sim, pd.DataFrame):
            x_sim = to_torch(x_sim.values)
        elif isinstance(x_sim, torch.Tensor):
            x_sim = x_sim.to(dtype=torch.get_default_dtype())

        with torch.no_grad():
            sim_x_scaled = self.scaler.scale(x_sim)
            # Getting the predictive distribution
            predictive_dist = self(sim_x_scaled)

            if type == 'f':
                # Getting samples from the GP (prior or posterior)
                realizations = predictive_dist.rsample()
            elif type == 'y':
                # Getting samples from the likelihood (observations)
                realizations = self.likelihood(predictive_dist).rsample()

            if return_type == "numpy":
                realizations = to_numpy(realizations)
            elif return_type == "torch":
                pass
            else:
                raise ValueError("Invalid return_type. Valid options are 'numpy' and 'torch'.")
            return realizations

    def make_predictions(self, test_x, type="f", return_type="numpy", posterior=True):
        """
        Make predictions using the fitted GPR model.

        Args:
            - test_x (Tensor): The test data inputs.

        Returns:
            - predictions (Tensor): The predictions for the test data.
        """

        if posterior is True:
            # GP regression predictions assume posterior
            self.eval()
            self.likelihood.eval()

        else:
            # Prior mode
            self.train()
            self.likelihood.train()

        if isinstance(test_x, np.ndarray):
            test_x = to_torch(test_x)
        elif isinstance(test_x, pd.DataFrame):
            test_x = to_torch(test_x.values)
        elif isinstance(test_x, torch.Tensor):
            test_x = test_x.to(dtype=torch.get_default_dtype())

        test_x_scaled = self.scaler.scale(test_x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type == "f":
                predictions = self(test_x_scaled)
            elif type == "y":
                predictions = self.likelihood(self(test_x_scaled))
            else:
                raise ValueError("Invalid type. Use 'f' for latent function or 'y' for noisy predictions.")
            self.predictions = Predictions(predictions.mean, predictions.variance)

        if return_type == "numpy":
            self.predictions.to_numpy()
        elif return_type == "torch":
            pass
        else:
            raise ValueError("Invalid return_type. Valid options are 'numpy' and 'torch'.")
        return self.predictions

    def get_LOOCV(self):
        self.train()

        y_dist = self.likelihood(self(self.train_x_scaled))
        K = y_dist.covariance_matrix
        y = self.train_y

        K_inv = torch.inverse(K)
        Ky_inv_product = torch.matmul(K_inv, y.unsqueeze(-1))

        K_inv_diagonal = torch.diag(K_inv)

        y_loocv = y - Ky_inv_product.squeeze() / K_inv_diagonal

        LOOCV_rmse = (y - y_loocv).pow(2).mean().sqrt()

        return LOOCV_rmse

    def compute_bic(self, data=None):
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted GPR model.

        Args:
            - data (Tensor): The data to use for computing the BIC. Uses training data by default.

        Returns:
            - bic (float): The BIC value for the fitted model.
        """
        # Implement the BIC computation procedure here

        self.train()

        if data is None:
            data = self.train_x_scaled

        # Get the number of data points
        n = data.shape[0]

        # Initialize the marginal log likelihood object
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self).to(data.device)

        # Get the model output for the data
        output = self(data)

        # Compute the log marginal likelihood
        log_marginal_likelihood = mll(output, self.train_y)

        # Get the number of hyperparameters
        self.num_param = sum(p[1].numel() for p in self.named_parameters())

        # Compute the BIC
        with torch.no_grad():
            self.bic = -log_marginal_likelihood * n + self.num_param * np.log(n) / 2

        return self.bic

    def get_hyperparameters_df(self, scaled=True):
        # TODO: beta1_orig = beta1 / (x_max - x_min); beta0_orig = beta0 - beta1_orig * x_min
        data = []
        for name, param in self.named_hyperparameters():
            # Getting the constraint corresponding to the current hyperparameter
            constraint = self.constraint_for_parameter_name(name)

            name = strip_raw_prefix(name)

            # Getting the transformed value (according to GPyTorch)
            if constraint is not None:
                transformed_value = constraint.transform(param)
            else:
                transformed_value = param

            # If it is a lengthscale, apply min/max unscaling. Note: only works in 1d currently.
            if scaled is False:
                if 'lengthscale' in name:
                    transformed_value = transformed_value * (self.scaler.maxs - self.scaler.mins)
                else:
                    transformed_value = transformed_value  # For non-lengthscale parameters, no unscaling is applied

            transformed_numpy = to_numpy(transformed_value)

            # Preparing data for dataframe
            if transformed_numpy.size > 1:  # Case where the parameter is a vector
                for idx, value in enumerate(transformed_value.flatten()):

                    entry = {
                        "Hyperparameter Name": name,
                        "Estimate": transformed_numpy.flatten()[idx]
                    }

                    data.append(entry)
            else:  # Case where the parameter is a scalar
                entry = {
                    "Hyperparameter Name": name,
                    "Estimate": transformed_numpy.item()
                }

                data.append(entry)

        # Creating dataframe
        df = pd.DataFrame(data)
        return df


class Predictions:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def to_numpy(self):
        self.mean = to_numpy(self.mean)
        self.variance = to_numpy(self.variance)

    def to_torch(self):
        self.mean = to_torch(self.mean)
        self.variance = to_torch(self.variance)


def strip_raw_prefix(s):
    # Split the string on the dot to isolate the last segment
    parts = s.split('.')
    # Check if the last segment starts with 'raw_' and strip it if true
    if parts[-1].startswith('raw_'):
        parts[-1] = parts[-1][4:]  # Remove the first 4 characters 'raw_'
    # Join the parts back together
    return '.'.join(parts)
