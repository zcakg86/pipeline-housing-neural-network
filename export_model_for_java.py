"""
Export trained PyTorch model and all artifacts for Java/ONNX consumption.
Outputs to java-app/model-artifacts/
"""
import sys
sys.path.append('src')

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

OUTPUT_DIR = Path('java-app/model-artifacts')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Find latest model ──────────────────────────────────────────────────────
def find_latest_model():
    dirs = sorted(glob('outputs/models/*/model.pth'))
    if not dirs:
        raise FileNotFoundError("No trained model found in outputs/models/")
    return Path(dirs[-1]).parent

model_dir = find_latest_model()
print(f"Exporting model from: {model_dir}")


# ── 2. Load model via model manager ──────────────────────────────────────────
from pricemodel.model_manager import modelmanager

manager = modelmanager()
manager.load_model(model_dir)
# Move to CPU for ONNX export (MPS device not supported by ONNX tracer)
manager.predictor.model = manager.predictor.model.cpu()
manager.predictor.model.eval()
print(f"  Neighborhood pooling: {manager.use_neighborhood_pooling}")
print(f"  Pooling strategy:     {manager.pooling_strategy}")
print(f"  Communities:          {manager.n_communities}")


# ── 3. Export ONNX ────────────────────────────────────────────────────────────
print("\nExporting ONNX model...")

batch = 2  # Use batch > 1 so dynamic axes work
community_in = torch.zeros(batch, 7, dtype=torch.long)
year_in      = torch.zeros(batch, dtype=torch.long)
week_in      = torch.zeros(batch, dtype=torch.long)
prop_in      = torch.zeros(batch, manager.property_dim,        dtype=torch.float32)
time_in      = torch.zeros(batch, manager.continuous_time_dim, dtype=torch.float32)
market_in    = torch.zeros(batch, manager.market_dim,          dtype=torch.float32)

onnx_path = OUTPUT_DIR / 'model.onnx'


class ModelWithExtras(torch.nn.Module):
    """
    Thin wrapper that exposes three outputs for ONNX export:
      - log_price_scaled  : shape [batch, 1]   — scaled log price
      - log_var_scaled    : shape [batch, 1]   — log variance (uncertainty head)
      - cls_attention     : shape [batch, 6]   — CLS→token attention weights
                            token order: community, year, week, property, time, market
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, community_indices, year, week,
                property_features, time_features, market_features):
        log_price, log_var = self.model(
            community_indices, year, week,
            property_features, time_features, market_features,
            return_uncertainty=True
        )
        # last_cls_attention: [batch, n_heads, 6] → average over heads → [batch, 6]
        cls_attn = self.model.last_cls_attention.mean(dim=1)  # [batch, 6]
        return log_price, log_var, cls_attn


wrapper = ModelWithExtras(manager.predictor.model)
wrapper.eval()

torch.onnx.export(
    wrapper,
    (community_in, year_in, week_in, prop_in, time_in, market_in),
    str(onnx_path),
    input_names=['community_indices', 'year', 'week',
                 'property_features', 'time_features', 'market_features'],
    output_names=['log_price_scaled', 'log_var_scaled', 'cls_attention'],
    dynamic_axes={
        'community_indices': {0: 'batch'},
        'year':              {0: 'batch'},
        'week':              {0: 'batch'},
        'property_features': {0: 'batch'},
        'time_features':     {0: 'batch'},
        'market_features':   {0: 'batch'},
        'log_price_scaled':  {0: 'batch'},
        'log_var_scaled':    {0: 'batch'},
        'cls_attention':     {0: 'batch'},
    },
    opset_version=17
)
print(f"  ✓ Saved: {onnx_path}  (outputs: log_price_scaled, log_var_scaled, cls_attention[6])")


# ── 4. Export scalers as JSON ─────────────────────────────────────────────────
print("\nExporting scalers as JSON...")

scaler_features = [
    'sqft', 'sqft_lot', 'beds',
    'time_trend',   # continuous_time_dim=1 (sin/cos removed)
    'mortgage_rate', 'unemployment_rate',
    'log_price'
]

scalers_json = {}
for feat in scaler_features:
    if feat in manager.scalers:
        sc = manager.scalers[feat]
        scalers_json[feat] = {
            'mean': float(sc.mean_[0]),
            'scale': float(sc.scale_[0])
        }
    else:
        print(f"  Warning: scaler for '{feat}' not found")

with open(OUTPUT_DIR / 'scalers.json', 'w') as f:
    json.dump(scalers_json, f, indent=2)
print(f"  ✓ Saved: {OUTPUT_DIR}/scalers.json  ({len(scalers_json)} scalers)")


# ── 5. Export vocabularies ────────────────────────────────────────────────────
print("\nExporting vocabularies...")
# Year vocab
year_vocab_out = {}
for k, v in manager.year_vocab.items():
    year_vocab_out[str(k)] = int(v)

with open(OUTPUT_DIR / 'year_vocab.json', 'w') as f:
    json.dump(year_vocab_out, f, indent=2)

# Week vocab
week_vocab_out = {}
for k, v in manager.week_vocab.items():
    week_vocab_out[str(k)] = int(v)

with open(OUTPUT_DIR / 'week_vocab.json', 'w') as f:
    json.dump(week_vocab_out, f, indent=2)

print(f"  ✓ year_vocab.json      ({len(year_vocab_out)} entries)")
print(f"  ✓ week_vocab.json      ({len(week_vocab_out)} entries)")


# ── 6. Export community map and H3 neighbor map ───────────────────────────────
import shutil

print("\nExporting community map...")
community_map_src = Path('data/community_map.json')
if community_map_src.exists():
    shutil.copy(community_map_src, OUTPUT_DIR / 'community_map.json')
    with open(community_map_src) as f:
        cm = json.load(f)
    n_communities = max(cm.values()) + 1
    print(f"  ✓ Copied community_map.json ({len(cm)} H3 L9 hexes, {n_communities} communities)")
else:
    print("  Warning: data/community_map.json not found — Java app will not resolve communities")

print("\nExporting H3 L8 neighbor map...")
h3_neighbor_src = Path('data/h3_l8_neighbor_communities.json')
if h3_neighbor_src.exists():
    shutil.copy(h3_neighbor_src, OUTPUT_DIR / 'h3_l8_neighbor_communities.json')
    with open(h3_neighbor_src) as f:
        h3_map = json.load(f)
    print(f"  ✓ Copied h3_l8_neighbor_communities.json ({len(h3_map)} hexes)")
else:
    print("  Warning: h3_l8_neighbor_communities.json not found")


# ── 7. Export model metadata ──────────────────────────────────────────────────
print("\nExporting model metadata...")

ckpt = torch.load(model_dir / 'model.pth', map_location='cpu')
reference_date = ckpt.get('reference_date', None)

metadata = {
    'model_version': 'current',
    'embedding_dim': manager.embedding_dim,
    'hidden_dim': manager.hidden_dim,
    'property_dim': manager.property_dim,
    'continuous_time_dim': manager.continuous_time_dim,
    'market_dim': manager.market_dim,
    'n_communities': manager.n_communities,
    'year_length': manager.year_length,
    'week_length': manager.week_length,
    'use_neighborhood_pooling': manager.use_neighborhood_pooling,
    'pooling_strategy': manager.pooling_strategy,
    'reference_date': reference_date,
    'property_features': ['sqft', 'sqft_lot', 'beds'],
    'time_features': ['time_trend'],
    'market_features': ['mortgage_rate', 'unemployment_rate'],
    'input_order': [
        'community_indices (long[batch,7])',
        'year (long[batch])',
        'week (long[batch])',
        f'property_features (float[batch,{manager.property_dim}])',
        f'time_features (float[batch,{manager.continuous_time_dim}])',
        f'market_features (float[batch,{manager.market_dim}])'
    ],
    'output': 'log_price_scaled (float[batch,1]) - apply log_price scaler inverse then exp()'
}

with open(OUTPUT_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Saved: {OUTPUT_DIR}/model_metadata.json")


# ── 8. Verify ONNX output matches PyTorch ────────────────────────────────────
print("\nVerifying ONNX output matches PyTorch...")
try:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))

    # Run PyTorch
    with torch.no_grad():
        pt_price, pt_logvar, pt_attn = wrapper(
            community_in, year_in, week_in, prop_in, time_in, market_in
        )
        pt_out = pt_price.numpy()

    # Run ONNX
    onnx_price, onnx_logvar, onnx_attn = sess.run(None, {
        'community_indices': community_in.numpy(),
        'year':              year_in.numpy(),
        'week':              week_in.numpy(),
        'property_features': prop_in.numpy(),
        'time_features':     time_in.numpy(),
        'market_features':   market_in.numpy(),
    })

    max_diff = np.abs(pt_out - onnx_price).max()
    print(f"  Max difference PyTorch vs ONNX (price): {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  ✓ ONNX output matches PyTorch")
    else:
        print("  ⚠ Larger than expected difference - check model")
    print(f"  log_var range:    [{onnx_logvar.min():.3f}, {onnx_logvar.max():.3f}]")
    print(f"  cls_attention sum per row: {onnx_attn.sum(axis=1)}")  # should be ~1.0

except ImportError:
    print("  onnxruntime not installed, skipping verification")
    print("  Install with: pip install onnxruntime")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Export complete → {OUTPUT_DIR}/")
print(f"{'='*60}")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>8.1f} KB")
