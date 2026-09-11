"""Training lifecycle for the standalone monthly H3 GraphSAGE baseline."""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .feature_contract import MARKET_FEATURES, PROPERTY_FEATURES, TIME_FEATURES
from .evaluation import calculate_validation_community_metrics
from .gnn_data import MonthlyGraphData
from .gnn_explain import FEATURE_SHAPLEY_COALITIONS, explain_gnn_prediction
from .gnn_network import H3GraphPriceModel
from .reproducibility import seed_everything
from .run_manifest import build_run_manifest, write_run_manifest


GNN_ARCHITECTURE_VERSION = 2


@dataclass(frozen=True)
class GNNRunArtifacts:
    """Serializable details required to reproduce one GNN baseline run."""

    train_indices: list[int]
    validation_indices: list[int]
    cell_ids: list[str]
    month_starts: list[str]
    node_feature_names: list[str]
    property_feature_names: list[str]
    time_feature_names: list[str]
    market_feature_names: list[str]


class GNNBaselineTrainer:
    """Fit GraphSAGE using month-grouped, chronological sale batches.

    The graph is evaluated once per month batch, then the corresponding sales
    consume their cell embeddings.  This avoids repeating the same GNN forward
    pass for every individual sale while retaining a standard MSE objective.
    """

    def __init__(
        self,
        graph_data: MonthlyGraphData,
        *,
        graph_hidden_dim: int = 32,
        head_hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 3e-4,
        random_seed: int = 42,
        graph_layer_norm: bool = False,
        graph_residual: bool = False,
        directory_prefix: str = "outputs/gnn",
    ):
        self.graph_data = graph_data
        self.graph_hidden_dim = graph_hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.graph_layer_norm = bool(graph_layer_norm)
        self.graph_residual = bool(graph_residual)
        self.architecture_version = GNN_ARCHITECTURE_VERSION
        self.device = torch.device(
            "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = Path(directory_prefix) / self.timestamp
        self.scalers: dict[str, StandardScaler] = {}
        self.model: H3GraphPriceModel | None = None
        self.optimizer = None
        self.scheduler = None
        self.train_indices: np.ndarray | None = None
        self.validation_indices: np.ndarray | None = None
        # Compatibility name used by the shared validation-community evaluator.
        self.val_indices: np.ndarray | None = None
        self.results = {"train_losses": [], "val_losses": [], "timestamp": self.timestamp}

    def split_chronologically(self, train_ratio: float) -> None:
        """Split the already date-sorted sale frame without random row mixing."""
        total = len(self.graph_data.dataframe)
        split = int(total * train_ratio)
        if split == 0 or split == total:
            raise ValueError("Chronological GNN split must leave train and validation sales")
        self.train_indices = np.arange(split, dtype=np.int64)
        self.validation_indices = np.arange(split, total, dtype=np.int64)
        self.val_indices = self.validation_indices

    def _fit_scalers(self) -> None:
        if self.train_indices is None:
            raise RuntimeError("Call split_chronologically before fitting GNN scalers")
        frame = self.graph_data.dataframe
        groups = {
            "property": PROPERTY_FEATURES,
            "time": TIME_FEATURES,
            "market": MARKET_FEATURES,
        }
        for group, names in groups.items():
            scaler = StandardScaler().fit(frame.iloc[self.train_indices][list(names)])
            self.scalers[group] = scaler
        self.scalers["target"] = StandardScaler().fit(
            frame.iloc[self.train_indices][["log_price"]]
        )

        # Fit only snapshots used by the training sale rows.  A snapshot is
        # causal by construction; this also keeps validation scale statistics
        # out of the learned representation.
        train_months = np.unique(self.graph_data.sale_month_index[self.train_indices])
        node_train_values = self.graph_data.node_features[train_months].reshape(
            -1, self.graph_data.node_features.shape[-1]
        )
        self.scalers["node"] = StandardScaler().fit(node_train_values)

    def _scaled_tensors(self) -> None:
        frame = self.graph_data.dataframe
        self.node_features = torch.tensor(
            self.scalers["node"].transform(
                self.graph_data.node_features.reshape(-1, self.graph_data.node_features.shape[-1])
            ).reshape(self.graph_data.node_features.shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.property_features = torch.tensor(
            self.scalers["property"].transform(frame[list(PROPERTY_FEATURES)]),
            dtype=torch.float32,
            device=self.device,
        )
        self.time_features = torch.tensor(
            self.scalers["time"].transform(frame[list(TIME_FEATURES)]),
            dtype=torch.float32,
            device=self.device,
        )
        self.market_features = torch.tensor(
            self.scalers["market"].transform(frame[list(MARKET_FEATURES)]),
            dtype=torch.float32,
            device=self.device,
        )
        self.targets = torch.tensor(
            self.scalers["target"].transform(frame[["log_price"]]),
            dtype=torch.float32,
            device=self.device,
        )
        self.edge_index = self.graph_data.edge_index.to(self.device)
        self.sale_node_index = torch.tensor(self.graph_data.sale_node_index, dtype=torch.long, device=self.device)

    @staticmethod
    def _group_indices_by_month(indices: np.ndarray, sale_month_index: np.ndarray) -> dict[int, np.ndarray]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for row in indices.tolist():
            grouped[int(sale_month_index[row])].append(row)
        return {month: np.asarray(rows, dtype=np.int64) for month, rows in grouped.items()}

    def _run_epoch(self, indices: np.ndarray, *, training: bool, generator: np.random.Generator) -> float:
        monthly = self._group_indices_by_month(indices, self.graph_data.sale_month_index)
        months = list(monthly)
        if training:
            generator.shuffle(months)
            self.model.train()
        else:
            self.model.eval()
        squared_error = 0.0
        sample_count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for month in months:
                rows = torch.tensor(monthly[month], dtype=torch.long, device=self.device)
                prediction = self.model(
                    self.node_features[month],
                    self.edge_index,
                    self.sale_node_index[rows],
                    self.property_features[rows],
                    self.time_features[rows],
                    self.market_features[rows],
                )
                error = prediction - self.targets[rows]
                loss = torch.mean(error.square())
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                squared_error += float(error.square().sum().detach().cpu())
                sample_count += len(rows)
        return squared_error / max(sample_count, 1)

    def fit(
        self,
        *,
        train_ratio: float = 0.7,
        epochs: int = 30,
        patience: int = 8,
        lr_plateau_factor: float = 0.5,
        lr_plateau_patience: int = 2,
        min_learning_rate: float = 1e-6,
    ) -> "GNNBaselineTrainer":
        """Train MSE price predictions and restore the best validation state."""
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must be positive")
        seed_everything(self.random_seed)
        self.split_chronologically(train_ratio)
        self.train_ratio = train_ratio
        self._fit_scalers()
        self._scaled_tensors()
        self.model = H3GraphPriceModel(
            node_feature_dim=self.graph_data.node_features.shape[-1],
            property_dim=len(PROPERTY_FEATURES),
            time_dim=len(TIME_FEATURES),
            market_dim=len(MARKET_FEATURES),
            graph_hidden_dim=self.graph_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
            dropout_rate=self.dropout_rate,
            graph_layer_norm=self.graph_layer_norm,
            graph_residual=self.graph_residual,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
            min_lr=min_learning_rate,
        )
        rng = np.random.default_rng(self.random_seed)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        for epoch in range(epochs):
            train_loss = self._run_epoch(self.train_indices, training=True, generator=rng)
            val_loss = self._run_epoch(self.validation_indices, training=False, generator=rng)
            self.scheduler.step(val_loss)
            self.results["train_losses"].append(train_loss)
            self.results["val_losses"].append(val_loss)
            self.results.setdefault("learning_rates", []).append(self.optimizer.param_groups[0]["lr"])
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train MSE: {train_loss:.4f}, "
                f"Val MSE: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.results["best_epoch"] = epoch + 1
                self.results["best_val_loss"] = val_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        if best_state is None:
            raise RuntimeError("GNN training finished without a valid validation checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def add_predictions_and_metrics(
        self,
        *,
        community_min_support: int = 20,
        community_shrinkage_strength: float = 20.0,
    ) -> pd.DataFrame:
        """Attach predictions and report GNN metrics using the neural conventions.

        Whole-dataset MAPE is retained for map-style comparison.  Validation
        MAPE/MSE and community reliability metrics remain chronological,
        held-out measures and are the quantities to use for model selection.
        """
        if self.model is None or self.validation_indices is None:
            raise RuntimeError("Train the GNN before generating evaluation metrics")
        self.model.eval()
        predictions = np.empty(len(self.graph_data.dataframe), dtype=np.float64)
        all_rows = np.arange(len(predictions), dtype=np.int64)
        grouped = self._group_indices_by_month(all_rows, self.graph_data.sale_month_index)
        with torch.no_grad():
            for month, rows_array in grouped.items():
                rows = torch.tensor(rows_array, dtype=torch.long, device=self.device)
                scaled = self.model(
                    self.node_features[month],
                    self.edge_index,
                    self.sale_node_index[rows],
                    self.property_features[rows],
                    self.time_features[rows],
                    self.market_features[rows],
                ).detach().cpu().numpy()
                predictions[rows_array] = scaled.reshape(-1)

        frame = self.graph_data.dataframe
        predicted_log_price = self.scalers["target"].inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(-1)
        frame["predicted_log_price"] = predicted_log_price
        frame["predicted_price"] = np.exp(predicted_log_price)
        frame["price_error"] = frame["predicted_price"] - frame["sale_price"]
        frame["pct_error"] = 100.0 * frame["price_error"] / frame["sale_price"]

        validation = frame.iloc[self.validation_indices]
        validation_log_error = validation["predicted_log_price"] - validation["log_price"]
        validation_price_error = validation["price_error"]
        metrics = {
            "whole_dataset_mape": float(frame["pct_error"].abs().mean()),
            "validation_mape": float(validation["pct_error"].abs().mean()),
            "validation_log_price_mse": float(np.mean(validation_log_error ** 2)),
            "validation_price_mse": float(np.mean(validation_price_error ** 2)),
            "validation_rmse": float(np.sqrt(np.mean(validation_price_error ** 2))),
            "validation_sample_count": int(len(validation)),
        }
        self.results.setdefault("metrics", {}).update(metrics)
        print(f"Whole-dataset MAPE: {metrics['whole_dataset_mape']:.2f}%")
        print(f"Validation MAPE: {metrics['validation_mape']:.2f}%")
        print(f"Validation log-price MSE: {metrics['validation_log_price_mse']:.4f}")
        print(
            f"Validation price RMSE: ${metrics['validation_rmse']:,.0f} "
            f"(MSE: ${metrics['validation_price_mse']:,.0f}²)"
        )

        if "community" in frame.columns and frame["community"].notna().any():
            self.dataframe = frame
            calculate_validation_community_metrics(
                self,
                min_support=community_min_support,
                shrinkage_strength=community_shrinkage_strength,
            )
        else:
            print("Skipping community metrics: no community-map evaluation labels found")
        return frame

    def explain_prediction(self, row_index: int, *, coalitions: int = FEATURE_SHAPLEY_COALITIONS):
        """Return exact-group and sampled feature Shapley effects for one sale."""
        return explain_gnn_prediction(self, row_index, coalitions=coalitions)

    @classmethod
    def load_for_explanations(cls, directory: str | Path) -> "GNNBaselineTrainer":
        """Load a saved GNN plus its monthly graph state for on-demand Shapley use."""
        directory = Path(directory)
        checkpoint = torch.load(directory / "gnn_model.pth", map_location="cpu")
        architecture_version = checkpoint.get("architecture_version")
        if architecture_version not in {1, 2}:
            raise ValueError("Unsupported GNN checkpoint architecture")
        graph_path = directory / "monthly_graph_state.npz"
        sales_path = directory / "sales_with_predictions.csv"
        if not graph_path.is_file() or not sales_path.is_file():
            raise FileNotFoundError(
                "This GNN run predates explanation artifacts. Re-run training with the "
                "current code to save monthly_graph_state.npz and sales_with_predictions.csv."
            )
        metadata = checkpoint["artifacts"]
        with np.load(graph_path, allow_pickle=False) as state:
            graph_data = MonthlyGraphData(
                dataframe=pd.read_csv(sales_path, parse_dates=["sale_date"]),
                cell_ids=tuple(metadata["cell_ids"]),
                month_starts=tuple(metadata["month_starts"]),
                edge_index=torch.tensor(state["edge_index"], dtype=torch.long),
                node_features=state["node_features"],
                sale_node_index=state["sale_node_index"],
                sale_month_index=state["sale_month_index"],
            )
        trainer = cls(
            graph_data,
            graph_hidden_dim=int(checkpoint["graph_hidden_dim"]),
            head_hidden_dim=int(checkpoint["head_hidden_dim"]),
            dropout_rate=float(checkpoint["dropout_rate"]),
            graph_layer_norm=bool(checkpoint.get("graph_layer_norm", False)),
            graph_residual=bool(checkpoint.get("graph_residual", False)),
        )
        trainer.architecture_version = architecture_version
        trainer.directory = directory
        trainer.results = checkpoint.get("results", {})
        trainer.train_indices = np.asarray(metadata["train_indices"], dtype=np.int64)
        trainer.validation_indices = np.asarray(metadata["validation_indices"], dtype=np.int64)
        trainer.val_indices = trainer.validation_indices
        for name in ("property", "time", "market", "target", "node"):
            trainer.scalers[name] = joblib.load(directory / f"{name}_scaler.pkl")
        trainer._scaled_tensors()
        trainer.model = H3GraphPriceModel(
            node_feature_dim=graph_data.node_features.shape[-1],
            property_dim=len(PROPERTY_FEATURES),
            time_dim=len(TIME_FEATURES),
            market_dim=len(MARKET_FEATURES),
            graph_hidden_dim=trainer.graph_hidden_dim,
            head_hidden_dim=trainer.head_hidden_dim,
            dropout_rate=trainer.dropout_rate,
            graph_layer_norm=trainer.graph_layer_norm,
            graph_residual=trainer.graph_residual,
        ).to(trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.eval()
        return trainer

    def save(self) -> Path:
        """Persist the best GNN weights and the exact causal graph lookup metadata."""
        if self.model is None or self.train_indices is None:
            raise RuntimeError("Train the GNN before saving it")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.architecture_version = GNN_ARCHITECTURE_VERSION
        artifacts = GNNRunArtifacts(
            train_indices=self.train_indices.tolist(),
            validation_indices=self.validation_indices.tolist(),
            cell_ids=list(self.graph_data.cell_ids),
            month_starts=list(self.graph_data.month_starts),
            node_feature_names=list(self.graph_data.node_feature_names),
            property_feature_names=list(PROPERTY_FEATURES),
            time_feature_names=list(TIME_FEATURES),
            market_feature_names=list(MARKET_FEATURES),
        )
        torch.save(
            {
                "architecture_version": GNN_ARCHITECTURE_VERSION,
                "model_state_dict": self.model.state_dict(),
                "graph_hidden_dim": self.graph_hidden_dim,
                "head_hidden_dim": self.head_hidden_dim,
                "dropout_rate": self.dropout_rate,
                "graph_layer_norm": self.graph_layer_norm,
                "graph_residual": self.graph_residual,
                "artifacts": asdict(artifacts),
                "results": self.results,
            },
            self.directory / "gnn_model.pth",
        )
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.directory / f"{name}_scaler.pkl")
        np.savez_compressed(
            self.directory / "monthly_graph_state.npz",
            node_features=self.graph_data.node_features,
            edge_index=self.graph_data.edge_index.detach().cpu().numpy(),
            sale_node_index=self.graph_data.sale_node_index,
            sale_month_index=self.graph_data.sale_month_index,
        )
        if "predicted_price" in self.graph_data.dataframe.columns:
            self.graph_data.dataframe.to_csv(
                self.directory / "sales_with_predictions.csv", index=False
            )
        training_config = self.results.get("training_config", {
            "train_ratio": getattr(self, "train_ratio", None),
            "random_seed": self.random_seed,
            "graph_hidden_dim": self.graph_hidden_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "graph_layer_norm": self.graph_layer_norm,
            "graph_residual": self.graph_residual,
        })
        write_run_manifest(self.directory, build_run_manifest(
            run_id=self.timestamp,
            model_family="gnn",
            architecture_version=self.architecture_version,
            architecture={
                "graph_layer_norm": self.graph_layer_norm,
                "graph_residual": self.graph_residual,
            },
            training_config=training_config,
            metrics=self.results.get("metrics", {}),
            train_indices=self.train_indices,
            validation_indices=self.validation_indices,
        ))
        return self.directory
