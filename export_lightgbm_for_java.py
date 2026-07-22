"""Export LightGBM to ONNX and append causal row predictions for Java."""

import argparse
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

from pricemodel.data_pipeline import DatasetBuilder
from pricemodel.deployment import atomic_deploy, create_staging_directory, write_manifest
from pricemodel.feature_contract import FEATURE_CONTRACT, write_feature_contract
from pricemodel.lightgbm_model import LightGBMPriceModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir")
    parser.add_argument("--sales", default="data/sales_2020_25.csv")
    parser.add_argument(
        "--historical-predictions",
        default="data/sales_2020_25_with_predictions.csv",
    )
    parser.add_argument(
        "--update-historical-predictions",
        action="store_true",
        help="Also atomically refresh the map-report CSV; deployment does not require it.",
    )
    parser.add_argument("--bundle-dir", help="Existing/new staging bundle directory")
    parser.add_argument("--deploy", action="store_true", help="Atomically install the complete bundle")
    return parser.parse_args(argv)


def numeric_feature_frame(frame, categorical_features):
    result = frame.copy()
    for column in categorical_features:
        result[column] = result[column].cat.codes
    return result.to_numpy(dtype=np.float32)


def export_lightgbm(args):
    """Export and verify LightGBM artifacts into one isolated bundle."""
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        candidates = sorted(Path("outputs/lightgbm").glob("*/lightgbm.joblib"))
        if not candidates:
            raise FileNotFoundError("No trained LightGBM model found in outputs/lightgbm")
        model_dir = candidates[-1].parent
    print(f"Exporting LightGBM model from {model_dir}")
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else create_staging_directory()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_dir / "lightgbm.joblib")
    model = bundle["model"]
    feature_metadata = bundle["feature_metadata"]
    feature_count = int(feature_metadata["feature_count"])

    prepared = DatasetBuilder()
    prepared._map_communities(pd.read_csv(args.sales))
    prepared._prepare_data(
        market_indicator_cache_path="data/market_indicators/fred_indicators.csv"
    )
    builder = LightGBMPriceModel()
    features = builder.build_feature_frame(prepared)
    expected_feature_names = list(model.feature_name_)
    missing_features = sorted(set(expected_feature_names).difference(features.columns))
    if missing_features:
        raise ValueError(
            f"Prepared data cannot satisfy the trained LightGBM contract: {missing_features}"
        )
    # An older deployed model may predate water features. Selecting its saved
    # feature order keeps that export path valid until the requested retrain.
    features = features[expected_feature_names]
    builder.categorical_features = [
        name for name in builder.categorical_features if name in features.columns
    ]
    predicted_log_price = model.predict(features)
    predicted_price = np.exp(predicted_log_price)
    actual_price = prepared.dataframe["sale_price"].to_numpy(dtype=np.float64)
    pct_error = (predicted_price - actual_price) / actual_price * 100.0

    historical_path = Path(args.historical_predictions)
    if args.update_historical_predictions:
        historical = pd.read_csv(historical_path)
        if len(historical) != len(prepared.dataframe):
            raise ValueError(
                f"Historical prediction rows ({len(historical)}) do not match "
                f"prepared rows ({len(prepared.dataframe)})"
            )
        for column in ["sale_price", "sqft", "sqft_lot"]:
            if not np.allclose(
                historical[column].to_numpy(dtype=float),
                prepared.dataframe[column].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise ValueError(f"Historical row order mismatch in {column}")
        historical["lightgbm_predicted_log_price"] = predicted_log_price
        historical["lightgbm_predicted_price"] = predicted_price
        historical["lightgbm_pct_error"] = pct_error
        historical_tmp = historical_path.with_suffix(historical_path.suffix + ".tmp")
        historical.to_csv(historical_tmp, index=False)
        os.replace(historical_tmp, historical_path)

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        target_opset=15,
    )
    temporary_onnx = bundle_dir / "lightgbm.onnx"
    onnxmltools.utils.save_model(onnx_model, temporary_onnx)

    # Compare ONNX and native LightGBM across the full chronology before export.
    check_indices = np.linspace(0, len(features) - 1, 512, dtype=int)
    check_frame = features.iloc[check_indices]
    check_values = numeric_feature_frame(check_frame, builder.categorical_features)
    session = ort.InferenceSession(
        str(temporary_onnx), providers=["CPUExecutionProvider"]
    )
    onnx_prediction = np.asarray(
        session.run(None, {"features": check_values})[0]
    ).reshape(-1)
    native_prediction = model.predict(check_frame)
    max_log_difference = float(
        np.max(np.abs(onnx_prediction - native_prediction))
    )
    if max_log_difference > 1e-4:
        raise ValueError(
            f"ONNX verification failed; max log-price difference {max_log_difference}"
        )

    validation_predictions = pd.read_csv(model_dir / "validation_predictions.csv")
    validation_log_residual = np.abs(
        np.log(validation_predictions["sale_price"].to_numpy(dtype=np.float64))
        - np.log(validation_predictions["predicted_price"].to_numpy(dtype=np.float64))
    )
    conformal_log_residual_95 = float(
        np.quantile(validation_log_residual, 0.95, method="higher")
    )

    metadata = {
        "model": "LGBMRegressor",
        "lightgbm_version": bundle["model"].booster_.params.get("version", "4.6.0"),
        "target": "log_price",
        "prediction_transform": "exp",
        "feature_count": feature_count,
        "feature_names": list(features.columns),
        "categorical_features": builder.categorical_features,
        "best_iteration": int(bundle["best_iteration"]),
        "onnx_input": f"features (float[batch,{feature_count}])",
        "onnx_output": "variable (float[batch,1], log_price)",
        "max_verified_log_price_difference": max_log_difference,
        "uncertainty": {
            "method": "held_out_absolute_log_residual_conformal",
            "coverage": 0.95,
            "calibration_sample_count": int(len(validation_predictions)),
            "absolute_log_residual_quantile": conformal_log_residual_95,
        },
    }
    metadata_path = bundle_dir / "lightgbm_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Keep the portable tree model for exact TreeSHAP and the ONNX model for
    # fast batch inference. Both must describe the generated shared contract.
    if list(features.columns) != FEATURE_CONTRACT["lightgbm_feature_names"]:
        raise ValueError("LightGBM model feature order differs from feature_contract.json")
    shutil.copy2(model_dir / "lightgbm_model.txt", bundle_dir / "lightgbm_model.txt")
    shutil.copy2("data/osm/king_county_water.geojson", bundle_dir / "king_county_water.geojson")
    shutil.copy2("data/market_indicators/fred_indicators.csv", bundle_dir / "fred_indicators.csv")
    write_feature_contract(bundle_dir / "feature_contract.json")
    write_manifest(bundle_dir, sources={"lightgbm_model_dir": str(model_dir)})
    if args.deploy:
        atomic_deploy(bundle_dir)

    if args.update_historical_predictions:
        print(f"Updated {historical_path} with {len(prepared.dataframe):,} predictions")
    print(f"Verified ONNX max log-price difference: {max_log_difference:.8f}")
    print(f"Exported LightGBM artifacts to: {bundle_dir}")
    return bundle_dir


def main(argv=None):
    return export_lightgbm(parse_args(argv))


if __name__ == "__main__":
    main()
