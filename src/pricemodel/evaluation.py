"""Evaluation and report-column generation for trained price models.

These functions deliberately operate on a ModelManager instance: they transform
predictions back to dollar space, attach diagnostics, and calculate validation-
only community reliability metrics without owning training or checkpoint state.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader


def add_predictions_to_data(
    manager,
    return_uncertainty=True,
    community_min_support=20,
    community_shrinkage_strength=20.0,
):
    """
    Add predictions with uncertainty estimates and CLS attention weights to dataframe.
    Uncertainty is always available regardless of training loss used.

    Community metrics are calculated only on the held-out validation rows.
    Raw estimates are retained for every community, while headline metrics
    use a minimum support threshold and empirical shrinkage toward overall
    validation MAPE.
    """
    loader = DataLoader(manager.tensors, batch_size=256)
    manager.predictor.eval()
    
    predictions  = []
    uncertainties = []
    targets       = []
    cls_attentions = []
    local_attentions = []
    local_gates = []
    local_residuals = []
    
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(manager.predictor.device) for t in batch)
            if manager.local_feature_dim > 0:
                community, property_feat, time_feat, market_feat, local_feat, target = batch
            else:
                community, property_feat, time_feat, market_feat, target = batch
                local_feat = None
            
            if return_uncertainty:
                # Uncertainty head is always built — available regardless of training loss
                pred, log_var = manager.predictor.model(
                    community, property_feat, time_feat, market_feat,
                    local_feat, return_uncertainty=True, need_weights=True
                )
                uncertainties.extend(torch.exp(log_var / 2).cpu().numpy())
            else:
                pred = manager.predictor.model(
                    community, property_feat, time_feat, market_feat,
                    local_feat, need_weights=True,
                )
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
            if manager.predictor.model.last_cls_attention is not None:
                cls_attn = manager.predictor.model.last_cls_attention.mean(dim=1).cpu().numpy()
                cls_attentions.extend(cls_attn)
            if manager.local_feature_dim > 0:
                local_attentions.extend(
                    manager.predictor.model.last_local_attention.cpu().numpy()
                )
                local_gates.extend(manager.predictor.model.last_local_gate.cpu().numpy())
                local_residuals.extend(
                    manager.predictor.model.last_local_residual.cpu().numpy()
                )
    
    # Reshape
    predictions = np.array(predictions).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)
    
    # Inverse transform
    scaler = manager.scalers['log_price']
    predicted_log_price = scaler.inverse_transform(predictions).ravel()
    target_log_price = scaler.inverse_transform(targets).ravel()
    
    # Add to dataframe
    manager.dataframe['predicted_log_price'] = predicted_log_price
    manager.dataframe['predicted_price'] = np.exp(predicted_log_price)
    manager.dataframe['target_log'] = target_log_price
    manager.dataframe['target'] = np.exp(target_log_price)
    
    # Add CLS attention weights
    # Tokens: [community, property, time, market]
    if len(cls_attentions) > 0:
        cls_attentions = np.array(cls_attentions)
        manager.dataframe['cls_attn_community'] = cls_attentions[:, 0]
        manager.dataframe['cls_attn_property'] = cls_attentions[:, 1]
        manager.dataframe['cls_attn_time'] = cls_attentions[:, 2]
        manager.dataframe['cls_attn_market'] = cls_attentions[:, 3]
        
        print(f"\nCLS Attention Weights (average across all predictions):")
        print(f"  Community: {manager.dataframe['cls_attn_community'].mean():.3f}")
        print(f"  Property:  {manager.dataframe['cls_attn_property'].mean():.3f}")
        print(f"  Time:      {manager.dataframe['cls_attn_time'].mean():.3f}")
        print(f"  Market:    {manager.dataframe['cls_attn_market'].mean():.3f}")

    if local_gates:
        local_gates = np.asarray(local_gates).reshape(-1)
        local_residuals = np.asarray(local_residuals).reshape(-1)
        local_attentions = np.asarray(local_attentions)
        manager.dataframe['local_residual_gate'] = local_gates
        manager.dataframe['local_residual_log_adjustment'] = (
            scaler.scale_[0] * local_gates * local_residuals
        )
        for position in range(7):
            manager.dataframe[f'local_attn_ring_{position}'] = local_attentions[:, position]
        print(
            f"Local residual gate mean: {local_gates.mean():.3f}; "
            f"mean absolute log adjustment: "
            f"{manager.dataframe['local_residual_log_adjustment'].abs().mean():.4f}"
        )
    
    # Uncertainty (in log space, then convert to price space)
    if len(uncertainties) > 0:
        uncertainties = np.array(uncertainties).reshape(-1, 1)
        uncertainty_unscaled = scaler.scale_[0] * np.array(uncertainties).ravel()
        manager.dataframe['prediction_std_log'] = uncertainty_unscaled
        # Approximate std in price space using delta method
        manager.dataframe['prediction_std_price'] = manager.dataframe['predicted_price'] * uncertainty_unscaled
        # 95% confidence interval
        manager.dataframe['price_lower_95'] = np.exp(predicted_log_price - 1.96 * uncertainty_unscaled)
        manager.dataframe['price_upper_95'] = np.exp(predicted_log_price + 1.96 * uncertainty_unscaled)
    
    # Error metrics
    manager.dataframe['price_error'] = manager.dataframe['predicted_price'] - manager.dataframe['sale_price']
    manager.dataframe['pct_error'] = 100 * (manager.dataframe['price_error'] / manager.dataframe['sale_price'])
    
    print(f'\nMean absolute percentage error: {manager.dataframe["pct_error"].abs().mean():.2f}%')

    calculate_validation_community_metrics(manager,
        min_support=community_min_support,
        shrinkage_strength=community_shrinkage_strength,
    )
    
    if len(uncertainties) > 0:
        # Calibration is an out-of-time metric. Whole-dataset coverage mixes
        # fitted training rows into the headline number and is misleading.
        in_ci = ((manager.dataframe['sale_price'] >= manager.dataframe['price_lower_95']) & 
                (manager.dataframe['sale_price'] <= manager.dataframe['price_upper_95']))
        if manager.val_indices is not None and len(manager.val_indices) > 0:
            validation_coverage = float(in_ci.iloc[manager.val_indices].mean() * 100)
            manager.results.setdefault('metrics', {})['validation_coverage_95'] = (
                validation_coverage
            )
            print(
                f'Validation 95% CI coverage: {validation_coverage:.1f}% '
                f'(target: 95%)'
            )
        else:
            print(f'Whole-dataset 95% CI coverage: {in_ci.mean()*100:.1f}%')
    
    return manager


def calculate_validation_community_metrics(
    manager,
    min_support=20,
    shrinkage_strength=20.0,
):
    """Save raw and reliability-adjusted community errors for validation."""
    if 'community' not in manager.dataframe.columns:
        return None
    if manager.val_indices is None or len(manager.val_indices) == 0:
        print(
            "Skipping community metrics: validation row indices are unavailable; "
            "call split_data() before add_predictions_to_data()."
        )
        return None
    if min_support < 1:
        raise ValueError("community_min_support must be at least 1")
    if shrinkage_strength < 0:
        raise ValueError("community_shrinkage_strength cannot be negative")

    evaluation = manager.dataframe.iloc[manager.val_indices].copy()
    evaluation['abs_pct_error'] = evaluation['pct_error'].abs()
    validation_mape = float(evaluation['abs_pct_error'].mean())

    summary = evaluation.groupby('community').agg(
        sales_count=('pct_error', 'size'),
        mean_abs_pct_error=('abs_pct_error', 'mean'),
        median_abs_pct_error=('abs_pct_error', 'median'),
        mean_pct_bias=('pct_error', 'mean'),
    )
    counts = summary['sales_count'].astype(float)
    summary['shrunk_mean_abs_pct_error'] = (
        counts * summary['mean_abs_pct_error'] +
        float(shrinkage_strength) * validation_mape
    ) / (counts + float(shrinkage_strength))
    summary['meets_min_support'] = summary['sales_count'] >= int(min_support)
    summary = summary.sort_values(
        'shrunk_mean_abs_pct_error', ascending=False
    )
    manager.community_error_summary = summary

    raw_mape = summary['mean_abs_pct_error']
    shrunk_mape = summary['shrunk_mean_abs_pct_error']
    supported_mape = summary.loc[
        summary['meets_min_support'], 'mean_abs_pct_error'
    ]

    metrics = {
        'validation_mape': validation_mape,
        'validation_sample_count': int(len(evaluation)),
        'validation_community_count': int(len(summary)),
        'community_min_support': int(min_support),
        'community_shrinkage_strength': float(shrinkage_strength),
        'community_supported_count': int(summary['meets_min_support'].sum()),
        'community_raw_mean_mape': float(raw_mape.mean()),
        'community_raw_p90_mape': float(raw_mape.quantile(0.9)),
        'community_raw_worst_mape': float(raw_mape.max()),
        'community_shrunk_mean_mape': float(shrunk_mape.mean()),
        'community_shrunk_p90_mape': float(shrunk_mape.quantile(0.9)),
        'community_shrunk_worst_mape': float(shrunk_mape.max()),
    }
    if not supported_mape.empty:
        metrics.update({
            'community_supported_mean_mape': float(supported_mape.mean()),
            'community_supported_p90_mape': float(supported_mape.quantile(0.9)),
            'community_supported_worst_mape': float(supported_mape.max()),
        })

    manager.results.setdefault('metrics', {}).update(metrics)
    os.makedirs(manager.directory, exist_ok=True)
    summary.to_csv(os.path.join(manager.directory, 'community_error_summary.csv'))

    print(
        f"Validation community MAPE — overall: {validation_mape:.2f}%, "
        f"supported communities: {len(supported_mape)}/{len(summary)} "
        f"(minimum {min_support} sales), "
        f"shrunk worst: {shrunk_mape.max():.2f}%"
    )
    if not supported_mape.empty:
        print(
            f"  Supported raw MAPE — mean: {supported_mape.mean():.2f}%, "
            f"p90: {supported_mape.quantile(0.9):.2f}%, "
            f"worst: {supported_mape.max():.2f}%"
        )
    return summary
