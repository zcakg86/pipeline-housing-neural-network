"""Sampled feature-level Shapley explanations for monthly H3 GNN predictions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import comb
from typing import Iterable

import numpy as np
import torch

from .feature_contract import MARKET_FEATURES, PROPERTY_FEATURES, TIME_FEATURES
from .local_market_features import LOCAL_MARKET_FEATURES


FEATURE_SHAPLEY_COALITIONS = 512
_GROUPS = ("Property", "Time", "Economics", "Cell market", "Spatial market")


@dataclass(frozen=True)
class GNNShapleyGroupEffect:
    """An exact Shapley effect for a logical GNN input group."""

    group: str
    log_contribution: float
    price_effect_pct: float


@dataclass(frozen=True)
class GNNShapleyFeatureEffect:
    """A sampled feature effect, projected to its exact group contribution."""

    feature: str
    group: str
    value: float
    affected_cell_count: int
    log_contribution: float
    price_effect_pct: float
    sampling_std_error_log: float
    sampling_std_error_pct: float


@dataclass(frozen=True)
class GNNShapleyExplanation:
    """A reconstruction-checked explanation in unscaled log-price space."""

    reference_log_price: float
    reference_price: float
    predicted_log_price: float
    predicted_price: float
    evaluated_group_coalitions: int
    sampled_feature_coalitions: int
    reference: str
    groups: tuple[GNNShapleyGroupEffect, ...]
    features: tuple[GNNShapleyFeatureEffect, ...]

    def as_dict(self) -> dict:
        """Return a JSON-ready representation compatible with map consumers."""
        result = asdict(self)
        result["groups"] = [asdict(item) for item in self.groups]
        result["features"] = [asdict(item) for item in self.features]
        return result


def _kernel_masks(feature_count: int, coalitions: int, seed: int) -> np.ndarray:
    """Return deterministic KernelSHAP masks, including empty/full anchors."""
    if coalitions < 4 or coalitions % 2:
        raise ValueError("coalitions must be an even integer of at least four")
    masks = np.zeros((coalitions, feature_count), dtype=bool)
    masks[1] = True
    rng = np.random.default_rng(seed)
    sizes = np.arange(1, feature_count)
    weights = np.asarray(
        [(feature_count - 1) / (comb(feature_count, int(size)) * size * (feature_count - size))
         for size in sizes],
        dtype=np.float64,
    )
    weights /= weights.sum()
    for row in range(2, coalitions, 2):
        size = int(rng.choice(sizes, p=weights))
        selected = rng.choice(feature_count, size=size, replace=False)
        masks[row, selected] = True
        masks[row + 1] = ~masks[row]
    return masks


def _kernel_regression(masks: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate KernelSHAP coefficients and their approximate standard errors."""
    feature_count = masks.shape[1]
    sizes = masks.sum(axis=1)
    weights = np.empty(len(masks), dtype=np.float64)
    for row, size in enumerate(sizes):
        if size == 0 or size == feature_count:
            weights[row] = 1_000_000.0
        else:
            weights[row] = (feature_count - 1) / (
                comb(feature_count, int(size)) * size * (feature_count - size)
            )
    design = np.column_stack([np.ones(len(masks)), masks.astype(np.float64)])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_values = values * np.sqrt(weights)
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
    residual = values - design @ coefficients
    degrees_of_freedom = max(len(values) - feature_count - 1, 1)
    residual_variance = float(np.sum(weights * residual ** 2) / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(weighted_design.T @ weighted_design)
    return coefficients[1:], np.sqrt(np.maximum(np.diag(covariance)[1:], 0.0))


def _two_hop_subgraph(trainer, target_node: int) -> tuple[np.ndarray, torch.Tensor, list[np.ndarray]]:
    """Extract precisely the two-hop receptive field of one GraphSAGE target."""
    edge = trainer.graph_data.edge_index.detach().cpu().numpy()
    adjacency: list[set[int]] = [set() for _ in trainer.graph_data.cell_ids]
    for source, target in edge.T:
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))
    hops: list[set[int]] = [{int(target_node)}]
    seen = set(hops[0])
    for _ in range(2):
        next_hop = set().union(*(adjacency[node] for node in hops[-1])) - seen
        hops.append(next_hop)
        seen.update(next_hop)
    node_ids = np.asarray(sorted(seen), dtype=np.int64)
    local_index = {int(node): index for index, node in enumerate(node_ids)}
    keep = np.isin(edge[0], node_ids) & np.isin(edge[1], node_ids)
    local_edge = np.asarray(
        [[local_index[int(source)], local_index[int(target)]] for source, target in edge[:, keep].T],
        dtype=np.int64,
    )
    local_edge_index = torch.tensor(
        local_edge.T if len(local_edge) else np.empty((2, 0), dtype=np.int64),
        dtype=torch.long,
        device=trainer.device,
    )
    return node_ids, local_edge_index, [
        np.asarray([local_index[node] for node in hop], dtype=np.int64) for hop in hops
    ]


def explain_gnn_prediction(
    trainer,
    row_index: int,
    *,
    coalitions: int = FEATURE_SHAPLEY_COALITIONS,
) -> GNNShapleyExplanation:
    """Explain one trained GNN prediction with exact groups and 512 sampled fields.

    Continuous inputs use their training means (scaled zero) as the reference.
    The spatial features are evaluated only over the target cell's two-hop
    receptive field, which is mathematically sufficient for this two-layer
    GraphSAGE network and avoids re-running the entire county graph.
    """
    if trainer.model is None:
        raise RuntimeError("Train or load a GNN model before explaining a prediction")
    if row_index < 0 or row_index >= len(trainer.graph_data.dataframe):
        raise IndexError(f"row_index must be between 0 and {len(trainer.graph_data.dataframe) - 1}")
    trainer.model.eval()
    row = trainer.graph_data.dataframe.iloc[row_index]
    month = int(trainer.graph_data.sale_month_index[row_index])
    target_node = int(trainer.graph_data.sale_node_index[row_index])
    node_ids, edge_index, hop_positions = _two_hop_subgraph(trainer, target_node)
    actual_nodes = trainer.node_features[month, node_ids].detach().cpu().numpy()
    target_position = int(np.where(node_ids == target_node)[0][0])

    feature_names: list[str] = []
    groups: list[int] = []
    values: list[float] = []
    counts: list[int] = []
    for group, names in enumerate((PROPERTY_FEATURES, TIME_FEATURES, MARKET_FEATURES)):
        for name in names:
            feature_names.append(name)
            groups.append(group)
            values.append(float(row[name]))
            counts.append(1)
    for field, name in enumerate(LOCAL_MARKET_FEATURES):
        feature_names.append(f"cell_{name}")
        groups.append(3)
        values.append(float(trainer.graph_data.node_features[month, target_node, field]))
        counts.append(1)
    for hop, prefix in ((1, "one_ring"), (2, "two_ring")):
        positions = hop_positions[hop]
        for field, name in enumerate(LOCAL_MARKET_FEATURES):
            feature_names.append(f"{prefix}_{name}")
            groups.append(4)
            values.append(float(np.mean(trainer.graph_data.node_features[month, node_ids[positions], field])) if len(positions) else 0.0)
            counts.append(int(len(positions)))
    feature_count = len(feature_names)
    seed_material = f"{row_index}|{row['h3_08']}|{row['sale_date']}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    masks = _kernel_masks(feature_count, coalitions, seed=seed)

    property_actual = trainer.property_features[row_index].detach().cpu().numpy()
    time_actual = trainer.time_features[row_index].detach().cpu().numpy()
    market_actual = trainer.market_features[row_index].detach().cpu().numpy()

    def evaluate(feature_mask: np.ndarray) -> float:
        position = 0
        property_values = np.where(feature_mask[position:position + len(PROPERTY_FEATURES)], property_actual, 0.0)
        position += len(PROPERTY_FEATURES)
        time_values = np.where(feature_mask[position:position + len(TIME_FEATURES)], time_actual, 0.0)
        position += len(TIME_FEATURES)
        market_values = np.where(feature_mask[position:position + len(MARKET_FEATURES)], market_actual, 0.0)
        position += len(MARKET_FEATURES)
        nodes = np.zeros_like(actual_nodes)
        for field in range(len(LOCAL_MARKET_FEATURES)):
            if feature_mask[position + field]:
                nodes[target_position, field] = actual_nodes[target_position, field]
        position += len(LOCAL_MARKET_FEATURES)
        for hop in (1, 2):
            node_positions = hop_positions[hop]
            for field in range(len(LOCAL_MARKET_FEATURES)):
                if feature_mask[position + field] and len(node_positions):
                    nodes[node_positions, field] = actual_nodes[node_positions, field]
            position += len(LOCAL_MARKET_FEATURES)
        with torch.no_grad():
            output = trainer.model(
                torch.tensor(nodes, dtype=torch.float32, device=trainer.device),
                edge_index,
                torch.tensor([target_position], dtype=torch.long, device=trainer.device),
                torch.tensor(property_values[None], dtype=torch.float32, device=trainer.device),
                torch.tensor(time_values[None], dtype=torch.float32, device=trainer.device),
                torch.tensor(market_values[None], dtype=torch.float32, device=trainer.device),
            )
        return float(output.item())

    # Exact five-group Shapley values provide reliable totals for the sampled
    # feature estimates, exactly as the Java neural explanation does.
    group_values = np.empty(1 << len(_GROUPS), dtype=np.float64)
    for group_mask in range(len(group_values)):
        include = np.asarray([(group_mask & (1 << group)) != 0 for group in groups], dtype=bool)
        group_values[group_mask] = evaluate(include)
    target_scale = float(trainer.scalers["target"].scale_[0])
    target_mean = float(trainer.scalers["target"].mean_[0])
    group_effects: list[GNNShapleyGroupEffect] = []
    for group, label in enumerate(_GROUPS):
        effect = 0.0
        player = 1 << group
        for coalition in range(len(group_values)):
            if coalition & player:
                continue
            size = coalition.bit_count()
            weight = 1.0 / (len(_GROUPS) * comb(len(_GROUPS) - 1, size))
            effect += weight * (group_values[coalition | player] - group_values[coalition])
        unscaled = target_scale * effect
        group_effects.append(GNNShapleyGroupEffect(label, unscaled, np.expm1(unscaled) * 100.0))

    sampled_values = np.asarray([evaluate(mask) for mask in masks], dtype=np.float64)
    contributions, standard_errors = _kernel_regression(masks, sampled_values)
    # Project the sampled effects onto exact group totals.  The final fields
    # therefore reconstruct the prediction despite sampling noise.
    for group in range(len(_GROUPS)):
        positions = np.flatnonzero(np.asarray(groups) == group)
        if not len(positions):
            continue
        correction = (
            group_effects[group].log_contribution / target_scale - contributions[positions].sum()
        ) / len(positions)
        contributions[positions] += correction
        if len(positions) == 1:
            standard_errors[positions] = 0.0
        else:
            standard_errors[positions] *= np.sqrt(1.0 - 1.0 / len(positions))
    feature_effects = tuple(
        GNNShapleyFeatureEffect(
            name, _GROUPS[group], value, count,
            float(target_scale * contribution), float(np.expm1(target_scale * contribution) * 100.0),
            float(target_scale * error), float(np.exp(target_scale * contribution) * target_scale * error * 100.0),
        )
        for name, group, value, count, contribution, error in zip(
            feature_names, groups, values, counts, contributions, standard_errors
        )
    )
    reference = target_mean + target_scale * group_values[0]
    predicted = target_mean + target_scale * group_values[-1]
    reconstructed = reference + sum(effect.log_contribution for effect in group_effects)
    if not np.isclose(reconstructed, predicted, atol=1e-5):
        raise RuntimeError("GNN Shapley group effects do not reconstruct the prediction")
    return GNNShapleyExplanation(
        reference, float(np.exp(reference)), predicted, float(np.exp(predicted)),
        len(group_values), len(masks),
        "training means for property, time, economics, and every causal graph-state field",
        tuple(group_effects), feature_effects,
    )
