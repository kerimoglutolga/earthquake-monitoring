#My own implementation basically from scratch

import collections
import enum
import math
import pathlib
import typing
import warnings
import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt
from PhaseNetPicker import PhaseNetPicker

class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2

class SWAGInference(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass all baselines by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        #DATASETUSE
        #train_xs: torch.Tensor,
        #model_dir: pathlib.Path,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 1,
        swag_learning_rate: float = 1e-6,
        swag_update_freq: int = 1,
        deviation_matrix_max_rank: int = 100,
        bma_samples: int = 5
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_epochs: Total number of gradient descent epochs for SWAG
        :param swag_learning_rate: Learning rate for SWAG gradient descent
        :param swag_update_freq: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param deviation_matrix_max_rank: Rank of deviation matrix for full SWAG
        :param bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        #self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_epochs = swag_epochs
        self.swag_learning_rate = swag_learning_rate
        self.swag_update_freq = swag_update_freq
        self.deviation_matrix_max_rank = deviation_matrix_max_rank
        self.bma_samples = bma_samples
        self.test_loader = test_loader
        self.train_loader = train_loader

        self.network = model
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.network.to(self.device)

        #SWAG init
        self.avg_first_moments = self._create_weight_copy()
        self.avg_second_moments = self._create_weight_copy()
        self.sigma_diag = self._create_weight_copy()
        self.deviation_matrix = self._create_weight_copy()
        for name, param in self.network.named_parameters():
            self.deviation_matrix[name] = collections.deque(maxlen=self.deviation_matrix_max_rank)

        # Calibration, prediction, and other attributes
        self._prediction_threshold = None  # this is an example, feel free to be creative
        

    def fit_swag(self, num_training_batches: int) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_learning_rate,
            momentum=0.01,
            nesterov=False,
            weight_decay=1e-9,
        )
        loss = torch.nn.MSELoss()

        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        # lr_scheduler = SWAGScheduler(
        #     optimizer,
        #     epochs=self.swag_epochs,
        #     steps_per_epoch=len(loader),
        # )

        # Perform initialization for SWAG fitting
        # Copy current network weights to initialize swag means & variance
        self.avg_first_moments = {name: param.detach().clone() 
                                for name, param in self.network.named_parameters()}
        self.avg_second_moments =  {name: param.detach().clone()**2 
                                    for name, param in self.network.named_parameters()}
        self.swag_iter_counter = 0

        self.network.train()
        with tqdm.trange(self.swag_epochs, desc="Running gradient descent for SWA") as pbar:
            pbar_dict = {}
            for epoch in pbar:

                average_loss = 0.0
                num_samples_processed = 0
                c = 0

                for waves, labels in self.train_loader:
                    if c > num_training_batches: break
                    c += 1
                    waves, labels = waves.to(self.device), labels.to(self.device)
                    outputs = self.network(waves).to(self.device)
                    optimizer.zero_grad()

                    # print("input device:", outputs.get_device())
                    # print("labels:", labels.to(torch.float32).get_device())

                    batch_loss = loss(input=outputs, target=labels)
                    batch_loss.backward()
                    optimizer.step()

                    #pbar_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    #lr_scheduler.step()
                    
                    # Calculate cumulative average training loss and accuracy
                    average_loss = (waves.size(0) * batch_loss.item() + num_samples_processed * average_loss) / \
                        (num_samples_processed + waves.size(0))

                    num_samples_processed += waves.size(0)
                    pbar_dict["avg. epoch loss"] = average_loss

                    pbar.set_postfix(pbar_dict)

                    # Update SWAG sample
                    if c % self.swag_update_freq == 0:
                        self.update_swag()
                        self.swag_iter_counter += 1
        
        # Compute diagonal matrix with moments running averages
        current_params = {name: param.detach().clone() for name, param in self.network.named_parameters()}
        for name, param in current_params.items():
            self.sigma_diag[name] = torch.clip(self.avg_second_moments[name] - self.avg_first_moments[name]**2, min=0)
        
    def update_swag(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        current_params = {name: param.detach().clone() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in current_params.items():
            # Update running average of moments
            self.avg_first_moments[name] = (self.avg_first_moments[name] * self.swag_iter_counter  + param) / (self.swag_iter_counter + 1)
            self.avg_second_moments[name] = (self.avg_second_moments[name] * self.swag_iter_counter  + param**2) / (self.swag_iter_counter + 1)
            if torch.isnan(self.avg_first_moments[name]).any():
                 print(f"In FITTING: avg first moments {name} contains NaN, Parameter")

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            for name, param in current_params.items():
                self.deviation_matrix[name].append(param -  self.avg_first_moments[name])

    def predict_probabilities_swag(self, num_eval_batches: int) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()
        per_model_sample_predictions = []

        # Perform Bayesian model averaging:
        # Instead of sampling self.bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.bma_samples networks,
        # and perform inference with each network on all samples in loader.

        if num_eval_batches == 1:

            for waves, _ in self.test_loader:
                waves = waves.to(self.device)
                break

        for _ in tqdm.trange(self.bma_samples, desc="Performing Bayesian model averaging"):
            # Sample new parameters for self.network from the SWAG approximate posterior
            self.sample_parameters()
            predictions = []
            k = 0
            preds = self.network(waves)
            predictions.append(preds)

            if num_eval_batches != 1:
                for waves, _ in self.test_loader:
                    waves = waves.to(self.device)
                    preds = self.network(waves)
                    predictions.append(preds)
                    k += 1
                    if k >= num_eval_batches:
                        break

            predictions = torch.cat(predictions)
            per_model_sample_predictions.append(predictions)

        bma_probabilities = torch.stack(per_model_sample_predictions, dim=0).mean(dim=0)
        
        return bma_probabilities, per_model_sample_predictions

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """  

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for name, param in self.network.named_parameters():

            # SWAG-diagonal part
            z_1 = torch.randn(param.size()).detach().clone().to(self.device)
            current_mean = self.avg_first_moments[name].detach().clone().to(self.device)
            current_std = torch.sqrt(torch.clip(self.avg_second_moments[name] - current_mean ** 2, min=0)).to(self.device)

            if torch.isnan(self.avg_first_moments[name]).any():
                print(f"Name: avg first moments {name} contains NaN, Parameter")

            if torch.isnan(current_mean).any():
                print(f"Name: current mean {name} contains NaN, Parameter")

            if torch.isnan(current_std).any():
                print(f"Name: current std {name} contains NaN, Parameter")

            if torch.isnan(self.sigma_diag[name].sqrt()).any():
                print(f"Name: sigma diag {name} contains NaN, Parameter")


            assert current_mean.size() == param.size() and current_std.size() == param.size()

            # Diagonal part
            if self.inference_mode == InferenceMode.SWAG_DIAGONAL:
                sampled_param = current_mean + 1./math.sqrt(2) * current_std * z_1
        
            # Full SWAG part
            if self.inference_mode == InferenceMode.SWAG_FULL:
                z_2 = torch.randn(min(self.deviation_matrix_max_rank, self.swag_iter_counter)).to(self.device)
                deviation_matrix = torch.stack(list(self.deviation_matrix[name]), dim=1).to(self.device)
                term1 = current_mean + 1/np.sqrt(2) * current_std * z_1
                term2 = 1/np.sqrt(2 * (self.deviation_matrix_max_rank - 1)) * torch.tensordot(deviation_matrix, z_2, dims=([1], [0]))
                sampled_param = term1 + term2

            if torch.isnan(sampled_param).any():
                print(f"Name: smapled param {name} contains NaN")

            # Modify weight value in-place; directly changing self.network
            # if name == "regression.weight": print(f"{name}, sampled weights: {sampled_param}"); print(f"{name}, original weights: {param.data}")

            assert param.data.shape == sampled_param.shape

            param.data = sampled_param.to(device=self.device, non_blocking=False)

        self._update_batchnorm()

    def _update_batchnorm(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.train_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        old_momentum_parameters = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            old_momentum_parameters[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        self.network.train()
        for i, (waves, _) in enumerate(self.train_loader):
            waves = waves.to(self.device)
            self.network(waves)
            if i > 200:
                break
        self.network.eval()
        

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }