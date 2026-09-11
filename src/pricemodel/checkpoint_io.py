"""Checkpoint persistence and exact round-trip verification.

ModelManager delegates serialization here so training orchestration is separate
from filesystem formats. Public functions accept a manager to preserve existing
checkpoint compatibility while providing a narrow persistence boundary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import torch

from .trainer import PriceTrainer
from .run_manifest import build_run_manifest, write_run_manifest


ATTENTION_ARCHITECTURE_VERSION = 5


def _atomic_torch_save(value, destination):
    """Replace a checkpoint only after the temporary file is complete."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value, destination):
    """Write, flush, and atomically replace a JSON artifact."""
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_results(manager):
    """Persist mutable metrics without rewriting the verified model weights."""
    Path(manager.directory).mkdir(parents=True, exist_ok=True)
    _atomic_json_dump(manager.results, Path(manager.directory) / "results.json")
    training_config = manager.results.get("training_config", {})
    write_run_manifest(manager.directory, build_run_manifest(
        run_id=Path(manager.directory).name,
        model_family="attention",
        architecture_version=getattr(
            manager, "architecture_version", ATTENTION_ARCHITECTURE_VERSION
        ),
        architecture={
            "attention_layer_norm": bool(getattr(manager, "attention_layer_norm", False)),
            "attention_residual": bool(getattr(manager, "attention_residual", False)),
            "use_local_correction": bool(
                getattr(manager, "use_local_correction", manager.local_feature_dim > 0)
            ),
        },
        training_config=training_config,
        metrics=manager.results.get("metrics", {}),
        train_indices=getattr(manager, "train_indices", None),
        validation_indices=getattr(manager, "val_indices", None),
    ))
    return manager


def synchronize_device(manager):
    """Finish queued accelerator work before checkpoint operations."""
    if manager.device.type == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif manager.device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(manager.device)


def save_and_reload_for_evaluation(manager):
    """Evaluate from the exact serialized best-checkpoint weights.

    The dataframe/tensors remain in memory, while split indices are
    preserved across load_model(), which normally treats a checkpoint as
    independent of any particular dataset split.
    """
    train_indices = (
        manager.train_indices.copy() if manager.train_indices is not None else None
    )
    val_indices = (
        manager.val_indices.copy() if manager.val_indices is not None else None
    )
    expected_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in manager.predictor.model.state_dict().items()
    }

    synchronize_device(manager)
    save_model(manager)
    load_model(manager, manager.directory)
    manager.train_indices = train_indices
    manager.val_indices = val_indices
    synchronize_device(manager)

    reloaded_state = manager.predictor.model.state_dict()
    mismatched = [
        name for name, expected in expected_state.items()
        if name not in reloaded_state or not torch.equal(
            expected, reloaded_state[name].detach().cpu()
        )
    ]
    if mismatched:
        raise RuntimeError(
            "Reloaded checkpoint does not match the trained best state: "
            + ", ".join(mismatched[:5])
        )

    manager.results['evaluation_checkpoint_reloaded'] = True
    print("Checkpoint round-trip verified; generating evaluation from reloaded weights.")
    return manager


def save_model(manager):
    """Atomically save weights and the immutable inference artifacts."""
    os.makedirs(manager.directory, exist_ok=True)

    synchronize_device(manager)
    manager.architecture_version = ATTENTION_ARCHITECTURE_VERSION

    if manager.use_neighborhood_pooling:
        manager.pooling_strategy = (
            manager.predictor.model.community_embedding.pooling_strategy
        )
    
    property_feature_names = list(manager._PROPERTY_FEATURES)
    if manager.property_dim == 3 and len(property_feature_names) != 3:
        # Compatibility for an explicitly constructed legacy predictor.
        property_feature_names = ['sqft', 'sqft_lot', 'beds']
    if manager.property_dim != len(property_feature_names):
        raise ValueError("Cannot save a model with an ambiguous property feature contract")

    checkpoint = {
        'architecture_version': ATTENTION_ARCHITECTURE_VERSION,
        'model_state_dict': manager.predictor.model.state_dict(),
        'optimizer_state_dict': manager.predictor.optimizer.state_dict(),
        'scheduler_state_dict': manager.predictor.scheduler.state_dict(),
        'results': manager.results,
        'embedding_dim': manager.embedding_dim,
        'community_embedding_dim': manager.community_embedding_dim,
        'hidden_dim': manager.hidden_dim,
        'property_dim': manager.property_dim,
        'property_feature_names': property_feature_names,
        'continuous_time_dim': manager.continuous_time_dim,
        'time_feature_names': list(manager._TIME_FEATURES),
        'market_dim': manager.market_dim,
        'n_communities': manager.n_communities,
        'community_embedding_length': (
            manager.n_communities if manager.use_neighborhood_pooling
            else manager.n_communities + 1
        ),
        'community_embedding_size': manager.n_communities + 1,
        'local_feature_dim': manager.local_feature_dim,
        'use_local_correction': bool(
            getattr(manager, 'use_local_correction', manager.local_feature_dim > 0)
        ),
        'local_feature_names': manager._LOCAL_FEATURES,
        'dropout_rate': manager.dropout_rate,
        'epochs': manager.epochs,
        'learning_rate': manager.learning_rate,
        'pooling_strategy': manager.pooling_strategy,
        'use_neighborhood_pooling': manager.use_neighborhood_pooling,
        'attention_layer_norm': bool(getattr(manager, 'attention_layer_norm', False)),
        'attention_residual': bool(getattr(manager, 'attention_residual', False)),
        'estimate_uncertainty': manager.estimate_uncertainty,
        'global_aux_weight': manager.global_aux_weight,
        'residual_penalty': manager.residual_penalty,
        'lr_plateau_factor': manager.lr_plateau_factor,
        'lr_plateau_patience': manager.lr_plateau_patience,
        'min_learning_rate': manager.min_learning_rate,
        'uncertainty_calibration_epochs': getattr(
            manager, 'uncertainty_calibration_epochs', 10
        ),
        'uncertainty_patience': getattr(manager, 'uncertainty_patience', 3),
        'random_seed': getattr(manager, 'random_seed', 42),
        'reference_date': manager.reference_date.isoformat() if manager.reference_date else None
    }
    _atomic_torch_save(checkpoint, Path(manager.directory) / "model.pth")
    
    save_results(manager)

    if manager.local_market_snapshot is not None:
        _atomic_json_dump(
            manager.local_market_snapshot,
            Path(manager.directory) / "local_market_snapshot.json",
        )
    if manager.neighbor_cells_map is not None:
        _atomic_json_dump(
            manager.neighbor_cells_map,
            Path(manager.directory) / "h3_l8_neighbor_cells.json",
        )
    
    print(f"Model saved to {manager.directory}")
    print(f"  Using neighborhood pooling: {manager.use_neighborhood_pooling}")
    if manager.use_neighborhood_pooling:
        print(f"  Pooling strategy: {manager.pooling_strategy}")
    
    return manager


def load_model(manager, directory):
    """Load saved model and artifacts"""
    from pathlib import Path
    directory = Path(directory)
    manager.directory = directory
    
    # Load checkpoint
    ckpt = torch.load(directory / "model.pth", map_location=manager.device)
    architecture_version = ckpt.get('architecture_version')
    if architecture_version not in {4, ATTENTION_ARCHITECTURE_VERSION}:
        raise ValueError(
            "This checkpoint predates the current compact-community and "
            "water-proximity-only architecture. "
            "Retrain with the current feature and model contract."
        )
    manager.architecture_version = architecture_version
    
    # Restore architecture params
    manager.embedding_dim = ckpt['embedding_dim']
    manager.community_embedding_dim = ckpt['community_embedding_dim']
    manager.hidden_dim = ckpt['hidden_dim']
    manager.property_dim = ckpt['property_dim']
    manager._PROPERTY_FEATURES = ckpt.get(
        'property_feature_names', ['sqft', 'sqft_lot', 'beds']
    )
    if manager.property_dim != len(manager._PROPERTY_FEATURES):
        raise ValueError(
            "Checkpoint property_dim does not match property_feature_names"
        )
    manager.continuous_time_dim = ckpt['continuous_time_dim']
    manager._TIME_FEATURES = ckpt.get(
        'time_feature_names', ['time_trend', 'annual_sin', 'annual_cos']
    )
    if manager.continuous_time_dim != len(manager._TIME_FEATURES):
        raise ValueError(
            "Checkpoint continuous_time_dim does not match time_feature_names"
        )
    manager.market_dim = ckpt['market_dim']
    manager.n_communities = ckpt['n_communities']
    manager.learning_rate = ckpt['learning_rate']
    manager.pooling_strategy = ckpt.get('pooling_strategy', 'mean')
    valid_pooling_strategies = {'mean', 'center_weighted', 'learnable'}
    if manager.pooling_strategy not in valid_pooling_strategies:
        raise ValueError(
            f"Checkpoint has invalid pooling strategy "
            f"'{manager.pooling_strategy}'"
        )
    manager.use_neighborhood_pooling = ckpt.get('use_neighborhood_pooling', False)
    manager.attention_layer_norm = bool(ckpt.get('attention_layer_norm', False))
    manager.attention_residual = bool(ckpt.get('attention_residual', False))
    manager.local_feature_dim = ckpt.get('local_feature_dim', 0)
    manager.use_local_correction = bool(
        ckpt.get('use_local_correction', manager.local_feature_dim > 0)
    )
    if manager.use_local_correction != (manager.local_feature_dim > 0):
        raise ValueError(
            "Checkpoint use_local_correction conflicts with local_feature_dim"
        )
    manager.dropout_rate = ckpt.get('dropout_rate', 0.1)
    manager.epochs = ckpt.get('epochs', 1)
    manager.estimate_uncertainty = ckpt.get('estimate_uncertainty', False)
    # Older checkpoints retain their original training configuration;
    # subsequent train_model() calls adopt the new stronger defaults unless
    # explicitly overridden.
    manager.global_aux_weight = ckpt.get('global_aux_weight', 0.1)
    manager.residual_penalty = ckpt.get('residual_penalty', 1e-3)
    manager.lr_plateau_factor = ckpt.get('lr_plateau_factor', 0.5)
    manager.lr_plateau_patience = ckpt.get('lr_plateau_patience', 3)
    manager.min_learning_rate = ckpt.get('min_learning_rate', 1e-6)
    manager.uncertainty_calibration_epochs = ckpt.get(
        'uncertainty_calibration_epochs', 10
    )
    manager.uncertainty_patience = ckpt.get('uncertainty_patience', 3)
    manager.random_seed = ckpt.get('random_seed', 42)
    manager.train_indices = None
    manager.val_indices = None
    
    if ckpt.get('reference_date'):
        manager.reference_date = pd.to_datetime(ckpt['reference_date'])
    
    # Recreate predictor
    # community_embedding_size = n_communities + 1 (includes unknown slot).
    # Fall back to n_communities + 1 for older checkpoints that didn't save this key.
    community_embedding_length = ckpt.get(
        'community_embedding_length',
        ckpt.get('community_embedding_size', manager.n_communities + 1)
    )
    manager.predictor = PriceTrainer(
        manager.device, manager.embedding_dim, manager.hidden_dim,
        manager.property_dim, manager.continuous_time_dim, manager.market_dim,
        community_embedding_length,
        manager.community_embedding_dim,
        manager.learning_rate,
        epochs=ckpt.get('epochs', 1),
        len_train_loader=1,
        dropout_rate=manager.dropout_rate,
        estimate_uncertainty=manager.estimate_uncertainty,
        use_neighborhood_pooling=manager.use_neighborhood_pooling,
        pooling_strategy=manager.pooling_strategy,
        local_feature_dim=manager.local_feature_dim,
        global_aux_weight=manager.global_aux_weight,
        residual_penalty=manager.residual_penalty,
        attention_layer_norm=manager.attention_layer_norm,
        attention_residual=manager.attention_residual,
        lr_plateau_factor=manager.lr_plateau_factor,
        lr_plateau_patience=manager.lr_plateau_patience,
        min_learning_rate=manager.min_learning_rate,
    )
    
    # Load weights
    manager.predictor.model.load_state_dict(ckpt['model_state_dict'])
    manager.predictor.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        manager.predictor.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    manager.predictor.eval()
    manager._synchronize_device()

    if manager.use_neighborhood_pooling:
        actual_strategy = (
            manager.predictor.model.community_embedding.pooling_strategy
        )
        if actual_strategy != manager.pooling_strategy:
            raise RuntimeError(
                "Loaded pooling strategy does not match checkpoint metadata: "
                f"{actual_strategy} != {manager.pooling_strategy}"
            )
    
    # Load scalers
    for feature in (
        manager._PROPERTY_FEATURES
        + manager._TIME_FEATURES
        + manager._MARKET_FEATURES
        + ['log_price']
        + manager._LOCAL_FEATURES
    ):
        scaler_path = directory / f"{feature}_scaler.pkl"
        if scaler_path.exists():
            manager.scalers[feature] = joblib.load(scaler_path)

    snapshot_path = directory / 'local_market_snapshot.json'
    if snapshot_path.exists():
        with open(snapshot_path, 'r') as f:
            manager.local_market_snapshot = json.load(f)
    neighbor_cells_path = directory / 'h3_l8_neighbor_cells.json'
    if neighbor_cells_path.exists():
        with open(neighbor_cells_path, 'r') as f:
            manager.neighbor_cells_map = json.load(f)
    
    # Load results
    manager.results = ckpt.get("results", {})
    
    print(f"Model loaded from {directory}")
    print(f"  Using neighborhood pooling: {manager.use_neighborhood_pooling}")
    if manager.use_neighborhood_pooling:
        print(f"  Pooling strategy: {manager.pooling_strategy}")
    
    return manager
