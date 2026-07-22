"""Optimization, early stopping, and uncertainty calibration for the neural model."""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .network import EnhancedEmbeddingModel

class PriceTrainer:
    """Optimize the price network and restore its best validation checkpoint.

    ``estimate_uncertainty=False`` minimizes the price mean with MSE only.
    When enabled, training first fits that mean with MSE and then freezes the
    price path while calibrating ``log_var`` with Gaussian NLL. The network can
    emit ``log_var`` in either mode, but it is meaningful only after calibration.
    """
    
    def __init__(self, device, embedding_dim, hidden_dim, property_dim,
                 continuous_time_dim, market_dim,
                 community_embedding_length,
                 year_length, week_length, learning_rate,
                 epochs, len_train_loader,
                 dropout_rate=0.1, estimate_uncertainty=False,
                 use_neighborhood_pooling=False, pooling_strategy='mean',
                 local_feature_dim=0, global_aux_weight=0.5,
                 residual_penalty=1e-2, balance_community_loss=True,
                 lr_plateau_factor=0.5, lr_plateau_patience=3,
                 min_learning_rate=1e-6,
                 train_community_loss_weights=None,
                 val_community_loss_weights=None):

        self.device = device
        self.estimate_uncertainty = estimate_uncertainty  # training loss only
        self.use_neighborhood_pooling = use_neighborhood_pooling
        self.local_feature_dim = int(local_feature_dim)
        self.global_aux_weight = global_aux_weight
        self.residual_penalty = residual_penalty
        self.balance_community_loss = balance_community_loss
        self.lr_plateau_factor = lr_plateau_factor
        self.lr_plateau_patience = lr_plateau_patience
        self.min_learning_rate = min_learning_rate
        self.train_community_loss_weights = None
        self.val_community_loss_weights = None
        self.best_epoch = None
        self.best_val_loss = None
        self.learning_rates = []
        
        self.model = EnhancedEmbeddingModel(
            embedding_dim, hidden_dim, property_dim,
            continuous_time_dim, market_dim,
            community_embedding_length, 
            year_length, week_length,
            dropout_rate=dropout_rate,
            use_neighborhood_pooling=use_neighborhood_pooling,
            pooling_strategy=pooling_strategy,
            local_feature_dim=self.local_feature_dim,
        ).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.configure_scheduler(
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=min_learning_rate,
        )
        self.set_community_loss_weights(
            train_community_loss_weights,
            val_community_loss_weights,
        )
    
    def eval(self):
        self.model.eval()

    def configure_scheduler(self, factor=0.5, patience=3, min_lr=1e-6):
        """Configure validation-driven learning-rate reduction."""
        self.lr_plateau_factor = factor
        self.lr_plateau_patience = patience
        self.min_learning_rate = min_lr
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=1e-4,
            threshold_mode='rel',
            min_lr=min_lr,
        )

    def set_community_loss_weights(self, train_weights=None, val_weights=None):
        """Install split-level inverse-frequency weights on the model device."""
        self.train_community_loss_weights = (
            train_weights.to(self.device) if train_weights is not None else None
        )
        self.val_community_loss_weights = (
            val_weights.to(self.device) if val_weights is not None else None
        )

    def _reduce_loss(self, per_sample_loss, community, split='train'):
        """Apply precomputed dataset-level equal-community weighting."""
        if not (self.local_feature_dim > 0 and self.balance_community_loss):
            return per_sample_loss.mean()
        center_community = community[:, 0] if community.ndim == 2 else community
        weights_table = (
            self.train_community_loss_weights
            if split == 'train'
            else self.val_community_loss_weights
        )
        if weights_table is None:
            return per_sample_loss.mean()
        sample_weights = weights_table[center_community]
        return (per_sample_loss * sample_weights).mean()
    
    def train_step(self, batch):
        """Single training step with optional uncertainty loss"""
        batch = tuple(t.to(self.device) for t in batch)
        if self.local_feature_dim > 0:
            community, year, week, property_feat, time_feat, market_feat, local_feat, targets = batch
        else:
            community, year, week, property_feat, time_feat, market_feat, targets = batch
            local_feat = None
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.estimate_uncertainty:
            predictions, log_var, components = self.model(
                community, year, week, property_feat, time_feat, market_feat,
                local_feat, return_uncertainty=True, return_components=True
            )
            # Negative log-likelihood (Gaussian).
            # Clamp log_var to a safe range to prevent:
            #   - log_var → -∞: precision → ∞, loss explodes
            #   - log_var →  ∞: loss dominated by variance term, ignores fit
            log_var = torch.clamp(log_var, min=-6.0, max=6.0)
            precision = torch.exp(-log_var)
            per_sample_loss = (
                precision.reshape(-1) * (predictions.reshape(-1) - targets.reshape(-1)) ** 2 +
                log_var.reshape(-1)
            )
            loss = self._reduce_loss(per_sample_loss, community)
        else:
            predictions, components = self.model(
                community, year, week, property_feat, time_feat, market_feat,
                local_feat, return_components=True
            )
            loss = self._reduce_loss(
                (predictions.reshape(-1) - targets.reshape(-1)) ** 2, community
            )

        if self.local_feature_dim > 0:
            global_loss = self._reduce_loss(
                (components["global_output"].reshape(-1) - targets.reshape(-1)) ** 2,
                community,
            )
            residual_size = torch.mean(components["local_delta"] ** 2)
            loss = loss + self.global_aux_weight * global_loss + self.residual_penalty * residual_size
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Keep the loss on-device. Calling item() here would synchronize MPS on
        # every batch and serialize otherwise asynchronous GPU work.
        return loss.detach()
    
    def train(self, train_loader, val_loader, epochs, patience=10):
        """Training loop with early stopping"""
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = None
        best_model_state = None
        best_optimizer_state = None
        best_scheduler_state = None
        learning_rates = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = torch.zeros((), device=self.device)
            for batch in train_loader:
                train_loss += self.train_step(batch)
            
            # Validation phase
            self.model.eval()
            val_loss = torch.zeros((), device=self.device)
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(t.to(self.device) for t in batch)
                    if self.local_feature_dim > 0:
                        community, year, week, property_feat, time_feat, market_feat, local_feat, targets = batch
                    else:
                        community, year, week, property_feat, time_feat, market_feat, targets = batch
                        local_feat = None
                    if self.estimate_uncertainty:
                        predictions, log_var, components = self.model(
                            community, year, week, property_feat, time_feat, market_feat,
                            local_feat, return_uncertainty=True, return_components=True
                        )
                        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
                        precision = torch.exp(-log_var)
                        per_sample_loss = (
                            precision.reshape(-1) *
                            (predictions.reshape(-1) - targets.reshape(-1)) ** 2 +
                            log_var.reshape(-1)
                        )
                        loss = self._reduce_loss(
                            per_sample_loss, community, split='val'
                        )
                    else:
                        predictions, components = self.model(
                            community, year, week, property_feat, time_feat, market_feat,
                            local_feat, return_components=True
                        )
                        loss = self._reduce_loss(
                            (predictions.reshape(-1) - targets.reshape(-1)) ** 2,
                            community,
                            split='val',
                        )
                    if self.local_feature_dim > 0:
                        global_loss = self._reduce_loss(
                            (components["global_output"].reshape(-1) -
                             targets.reshape(-1)) ** 2,
                            community,
                            split='val',
                        )
                        residual_size = torch.mean(components["local_delta"] ** 2)
                        loss = (loss + self.global_aux_weight * global_loss +
                                self.residual_penalty * residual_size)
                    val_loss += loss.detach()
            
            # Exactly one synchronization per aggregate at epoch end.
            avg_train_loss = (train_loss / len(train_loader)).item()
            avg_val_loss = (val_loss / len(val_loader)).item()
            if not np.isfinite(avg_train_loss) or not np.isfinite(avg_val_loss):
                raise FloatingPointError(
                    "Non-finite epoch loss detected; aborting before the "
                    "checkpoint is saved."
                )
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            previous_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            print(
                f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}'
            )
            if current_lr < previous_lr:
                print(
                    f'  ReduceLROnPlateau: learning rate '
                    f'{previous_lr:.2e} -> {current_lr:.2e}'
                )
            
            # Retain the complete state associated with the best validation
            # epoch, not merely the final state before patience expires.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(self.scheduler.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.optimizer.load_state_dict(best_optimizer_state)
            self.scheduler.load_state_dict(best_scheduler_state)
            print(
                f"Restored best checkpoint from epoch {best_epoch} "
                f"(validation loss {best_val_loss:.4f})"
            )

        self.best_epoch = best_epoch
        self.best_val_loss = best_val_loss
        self.learning_rates = learning_rates
        return train_losses, val_losses

    def _unpack_batch(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        if self.local_feature_dim > 0:
            community, year, week, property_feat, time_feat, market_feat, local_feat, targets = batch
        else:
            community, year, week, property_feat, time_feat, market_feat, targets = batch
            local_feat = None
        return community, year, week, property_feat, time_feat, market_feat, local_feat, targets

    def _diagnostic_vector(self, predictions, log_var, components, targets):
        """Return device-side diagnostic sums; synchronize only at epoch end."""
        residual = predictions.reshape(-1) - targets.reshape(-1)
        clipped_log_var = torch.clamp(log_var.reshape(-1), min=-6.0, max=6.0)
        nll = torch.exp(-clipped_log_var) * residual.square() + clipped_log_var
        predicted_std = torch.sqrt(torch.exp(clipped_log_var))
        global_residual = components["global_output"].reshape(-1) - targets.reshape(-1)
        local_adjustment = (
            components["local_gate"].reshape(-1) *
            components["local_delta"].reshape(-1)
        )
        return torch.stack([
            residual.square().sum(),
            nll.sum(),
            clipped_log_var.sum(),
            (residual.abs() <= 1.96 * predicted_std).to(torch.float32).sum(),
            global_residual.square().sum(),
            local_adjustment.abs().sum(),
            torch.tensor(float(residual.numel()), device=self.device),
        ]).detach()

    @staticmethod
    def _finalize_diagnostics(total):
        values = total.detach().cpu().numpy()
        count = max(values[6], 1.0)
        return {
            'prediction_mse': float(values[0] / count),
            'nll': float(values[1] / count),
            'mean_log_var': float(values[2] / count),
            'coverage_95': float(values[3] / count),
            'global_head_mse': float(values[4] / count),
            'mean_abs_residual_adjustment': float(values[5] / count),
        }

    def _mean_train_step(self, batch):
        community, year, week, property_feat, time_feat, market_feat, local_feat, targets = (
            self._unpack_batch(batch)
        )
        self.optimizer.zero_grad(set_to_none=True)
        predictions, log_var, components = self.model(
            community, year, week, property_feat, time_feat, market_feat,
            local_feat, return_uncertainty=True, return_components=True,
        )
        loss = self._reduce_loss(
            (predictions.reshape(-1) - targets.reshape(-1)).square(), community
        )
        if self.local_feature_dim > 0:
            global_loss = self._reduce_loss(
                (components["global_output"].reshape(-1) - targets.reshape(-1)).square(),
                community,
            )
            residual_size = torch.mean(components["local_delta"].square())
            loss = loss + self.global_aux_weight * global_loss + self.residual_penalty * residual_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        return loss.detach(), self._diagnostic_vector(
            predictions, log_var, components, targets
        )

    def _uncertainty_train_step(self, batch):
        community, year, week, property_feat, time_feat, market_feat, local_feat, targets = (
            self._unpack_batch(batch)
        )
        self.optimizer.zero_grad(set_to_none=True)
        predictions, log_var, components = self.model(
            community, year, week, property_feat, time_feat, market_feat,
            local_feat, return_uncertainty=True, return_components=True,
        )
        clipped_log_var = torch.clamp(log_var.reshape(-1), min=-6.0, max=6.0)
        residual = predictions.reshape(-1) - targets.reshape(-1)
        loss = self._reduce_loss(
            torch.exp(-clipped_log_var) * residual.square() + clipped_log_var,
            community,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        return loss.detach(), self._diagnostic_vector(
            predictions, log_var, components, targets
        )

    def _evaluate_stage(self, loader, objective):
        self.model.eval()
        loss_total = torch.zeros((), device=self.device)
        diagnostic_total = torch.zeros(7, device=self.device)
        with torch.no_grad():
            for batch in loader:
                community, year, week, property_feat, time_feat, market_feat, local_feat, targets = (
                    self._unpack_batch(batch)
                )
                predictions, log_var, components = self.model(
                    community, year, week, property_feat, time_feat, market_feat,
                    local_feat, return_uncertainty=True, return_components=True,
                )
                residual = predictions.reshape(-1) - targets.reshape(-1)
                if objective == 'nll':
                    clipped_log_var = torch.clamp(log_var.reshape(-1), min=-6.0, max=6.0)
                    per_sample = torch.exp(-clipped_log_var) * residual.square() + clipped_log_var
                else:
                    per_sample = residual.square()
                loss_total += self._reduce_loss(
                    per_sample, community, split='val'
                ).detach()
                diagnostic_total += self._diagnostic_vector(
                    predictions, log_var, components, targets
                )
        return (
            (loss_total / len(loader)).item(),
            self._finalize_diagnostics(diagnostic_total),
        )

    def train_two_stage(
        self,
        train_loader,
        val_loader,
        mean_epochs,
        mean_patience=5,
        uncertainty_epochs=10,
        uncertainty_patience=3,
        learning_rate=3e-4,
    ):
        """Train price weights with MSE, then calibrate frozen uncertainty heads."""
        history = []
        train_losses = []
        val_losses = []
        learning_rates = []

        uncertainty_parameters = list(self.model.uncertainty_layer.parameters())
        if self.local_feature_dim > 0:
            uncertainty_parameters += list(self.model.local_uncertainty_layer.parameters())
        uncertainty_ids = {id(parameter) for parameter in uncertainty_parameters}

        # Stage 1: price fitting. Uncertainty heads are held fixed and cannot
        # distort the mean model through the NLL precision term.
        for parameter in self.model.parameters():
            parameter.requires_grad = id(parameter) not in uncertainty_ids
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=learning_rate,
        )
        self.configure_scheduler(
            factor=self.lr_plateau_factor,
            patience=self.lr_plateau_patience,
            min_lr=self.min_learning_rate,
        )
        best_mean_state = None
        best_mean_mse = float('inf')
        best_mean_epoch = None
        patience_counter = 0

        for epoch in range(mean_epochs):
            self.model.train()
            train_loss_total = torch.zeros((), device=self.device)
            train_diagnostic_total = torch.zeros(7, device=self.device)
            for batch in train_loader:
                loss, diagnostics = self._mean_train_step(batch)
                train_loss_total += loss
                train_diagnostic_total += diagnostics
            train_loss = (train_loss_total / len(train_loader)).item()
            train_diagnostics = self._finalize_diagnostics(train_diagnostic_total)
            val_objective, val_diagnostics = self._evaluate_stage(val_loader, 'mse')
            val_mse = val_diagnostics['prediction_mse']
            previous_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_mse)
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            train_losses.append(train_loss)
            val_losses.append(val_mse)
            history.append({
                'phase': 'mean', 'epoch': epoch + 1,
                'learning_rate': current_lr,
                'train_objective': train_loss,
                'val_objective': val_objective,
                'train': train_diagnostics,
                'validation': val_diagnostics,
            })
            print(
                f"Mean epoch [{epoch + 1}/{mean_epochs}], train objective: "
                f"{train_loss:.4f}, val MSE: {val_mse:.4f}, LR: {current_lr:.2e}"
            )
            if current_lr < previous_lr:
                print(f"  ReduceLROnPlateau: {previous_lr:.2e} -> {current_lr:.2e}")
            if val_mse < best_mean_mse:
                best_mean_mse = val_mse
                best_mean_epoch = epoch + 1
                best_mean_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= mean_patience:
                    print(f"Mean-stage early stopping at epoch {epoch + 1}")
                    break

        if best_mean_state is None:
            raise RuntimeError("Mean stage did not produce a finite checkpoint")
        self.model.load_state_dict(best_mean_state)
        print(
            f"Restored best mean checkpoint from epoch {best_mean_epoch} "
            f"(validation MSE {best_mean_mse:.4f})"
        )

        # Stage 2: freeze every price-producing parameter. With dropout disabled,
        # only the two log-variance heads can move.
        best_uncertainty_state = None
        best_val_nll = float('inf')
        best_uncertainty_epoch = None
        if self.estimate_uncertainty and uncertainty_epochs > 0:
            for parameter in self.model.parameters():
                parameter.requires_grad = id(parameter) in uncertainty_ids
            self.optimizer = torch.optim.AdamW(uncertainty_parameters, lr=learning_rate)
            self.configure_scheduler(
                factor=self.lr_plateau_factor,
                patience=max(1, min(self.lr_plateau_patience, uncertainty_patience - 1)),
                min_lr=self.min_learning_rate,
            )
            patience_counter = 0
            for epoch in range(uncertainty_epochs):
                self.model.eval()
                train_loss_total = torch.zeros((), device=self.device)
                train_diagnostic_total = torch.zeros(7, device=self.device)
                for batch in train_loader:
                    loss, diagnostics = self._uncertainty_train_step(batch)
                    train_loss_total += loss
                    train_diagnostic_total += diagnostics
                train_loss = (train_loss_total / len(train_loader)).item()
                train_diagnostics = self._finalize_diagnostics(train_diagnostic_total)
                val_objective, val_diagnostics = self._evaluate_stage(val_loader, 'nll')
                val_nll = val_diagnostics['nll']
                previous_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_nll)
                current_lr = self.optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                history.append({
                    'phase': 'uncertainty', 'epoch': epoch + 1,
                    'learning_rate': current_lr,
                    'train_objective': train_loss,
                    'val_objective': val_objective,
                    'train': train_diagnostics,
                    'validation': val_diagnostics,
                })
                print(
                    f"Uncertainty epoch [{epoch + 1}/{uncertainty_epochs}], train NLL: "
                    f"{train_diagnostics['nll']:.4f}, val NLL: {val_nll:.4f}, "
                    f"coverage: {100 * val_diagnostics['coverage_95']:.1f}%, "
                    f"LR: {current_lr:.2e}"
                )
                if current_lr < previous_lr:
                    print(f"  ReduceLROnPlateau: {previous_lr:.2e} -> {current_lr:.2e}")
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_uncertainty_epoch = epoch + 1
                    best_uncertainty_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= uncertainty_patience:
                        print(f"Uncertainty-stage early stopping at epoch {epoch + 1}")
                        break
            if best_uncertainty_state is not None:
                self.model.load_state_dict(best_uncertainty_state)
                print(
                    f"Restored best uncertainty checkpoint from epoch "
                    f"{best_uncertainty_epoch} (validation NLL {best_val_nll:.4f})"
                )

        # Restore a standard full-model optimizer contract for serialization and
        # optional future fine-tuning. Model weights remain the two-stage best.
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.configure_scheduler(
            factor=self.lr_plateau_factor,
            patience=self.lr_plateau_patience,
            min_lr=self.min_learning_rate,
        )
        self.best_epoch = best_mean_epoch
        self.best_val_loss = best_mean_mse
        self.best_uncertainty_epoch = best_uncertainty_epoch
        self.best_val_nll = best_val_nll if best_uncertainty_state is not None else None
        self.learning_rates = learning_rates
        self.diagnostic_history = history
        return train_losses, val_losses




# Compatibility alias retained for existing scripts and checkpoint-loading code.
price_predictor = PriceTrainer
