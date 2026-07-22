# Project: Transformer-Based Real Estate Price Prediction

### **Executive Summary**
This project implements an end-to-end deep learning pipeline designed to forecast real estate prices. It utilizes a custom **Attention-Based Neural Network** that treats tabular property data as a sequence of tokens, allowing the model to learn complex, non-linear relationships between location, time, and physical property characteristics.

### Aims
* Implement location-specific embedding through spatial features and community detection, that can estimate prices without bias across King County.
* Provide accurate estimates throughout time.
* Produce dashboard map to present model predictions and performance.


### Data
*   **Historical Residential Sales data:**\
Kaggle [King County Sales, Andy Krause](https://www.kaggle.com/datasets/andykrause/kingcountysales/data) Version 8: kingco_sales.csv\
Contains sales of single family homes in King County, sold between 1999 and year end 2025.
*   **Recent House Sales:**\
RentCast API properties endpoint.
*   **Zillow listings:**\
Fetched from HasData API.
*  **Macroeconomic indicators:**\
FED of St Louis: [FRED graph](https://fred.stlouisfed.org/graph/fredgraph.csv?id=") \
  MORTGAGE30US : 30-year fixed mortgage rate (weekly, %)\
  UNRATE       : US civilian unemployment rate (monthly, %)

### **Feature processing**
* H3 Index for spatial representation
* Louvain graph based community detection used for identification of location submarkets. 

### **Neural Network Architecture**
The core model utilizes a Transformer-inspired architecture adapted for structured tabular data:

*   **Input Representation:** The model ingests heterogeneous data types by projecting them into a shared latent embedding space.
*   **Categorical Embeddings:** Learnable vector representations for *Community* (Location), *Year*, and *Week* (Seasonality).
*   **Numeric Projections:** Linear projections map numerical groups — property features, continuous time features, and market indicators — into the same latent dimension before they are combined.
*   **Sequence Construction:** These vectors are stacked as tokens to form a sequence, prepended by a learnable **`[CLS]` token**.
*   **Self-Attention Mechanism:** A Multi-Head Attention layer enables every token to interact with every other token. The `[CLS]` token aggregates a global summary of the listing by attending to location, time, property, and market signals.
*   **Regression Head:** The final price prediction is generated from the contextualized `[CLS]` output via a Multi-Layer Perceptron (MLP).
*   **Uncertainty Output:** An optional uncertainty head is available and returns a log-variance estimate alongside the prediction.

### **Pipeline & Data Lifecycle**
The project includes a robust `ModelManager` framework that orchestrates the entire machine learning lifecycle:

*   **Data Ingestion & Engineering:** Automated cleaning, null-handling, and feature derivation (e.g., Log-Price transformation). It handles vocabulary creation for categorical mapping and `StandardScaler` fitting for numerical normalization.
*   **Training Loop:** A custom training cycle utilizing the Adam optimizer and Mean Squared Error (MSE) loss, featuring real-time validation monitoring and attention-weight tracking.
*   **Inference & Analysis:** A prediction engine that reverses transformations to output interpretable prices, calculates percentage errors per listing, and appends results to the original dataset.
*   **Artifact Management:** Automatic serialization of model weights, optimizer states, scalers, and vocabulary dictionaries to JSON and Pickle files, ensuring full reproducibility and easy deployment.

### **Technology Stack**
*   **Core Framework:** PyTorch (Neural Networks, Tensors)
*   **Data Manipulation:** Pandas, NumPy
*   **Preprocessing:** Scikit-Learn (StandardScaler)
*   **Serialization:** Joblib, Pickle, JSON
*   **Java Export:** `deploy_models_for_java.py` stages both models, verifies
    ONNX parity and the shared feature contract, writes a hash manifest, and
    atomically installs one Java artifact bundle.

For checkout-based development, commands use the `src` package layout:

```bash
python3 -m pip install -e .
# or prefix an individual command with PYTHONPATH=src
```

### **Java Integration**
See [java-app/house-price-app/README.md](https://github.com/zcakg86/pipeline-housing-neural-network/tree/kiraze/java-app/house-price-app) for the Java app and how the exported model is consumed.

### GitHub Codespaces

The `.devcontainer` configuration provides Python 3.11, Java 21, the pinned
Python environment, warmed Maven dependencies, and a private forwarded Quarkus
port. After the Codespace finishes its setup:

```bash
make test
make java-dev
```

Port 8080 appears as **Quarkus house-price app** in the Codespaces Ports panel.
See [.devcontainer/README.md](.devcontainer/README.md) for required model
artifacts, optional API secrets, and the boundary between Codespaces development
and production deployment.

### Random Forest baseline

`main_random_forest.py` trains a quick `RandomForestRegressor` baseline using
the same chronological 70/30 split and prepared inputs as the neural model.
Community, year, and week values are encoded categorically; the six neighboring
communities use a multi-hot representation, and all 7 x 5 leakage-safe local
market values are included.

```bash
python3 main_random_forest.py
```

Artifacts are written under `outputs/random_forest/<timestamp>/`, including the
Joblib model, validation-only metrics, and validation predictions.

### LightGBM baseline

`main_lightgbm.py` uses the same prepared inputs and final chronological 30%
holdout. Community, neighbor-community, year, and week values are passed as
native categorical features. Boosting rounds are selected using a chronological
slice inside the training period, after which the model is refit on the full 70%
training period before the final holdout is evaluated.

```bash
python3 main_lightgbm.py
```

On macOS, install LightGBM's OpenMP runtime once with `brew install libomp`.

Artifacts are written under `outputs/lightgbm/<timestamp>/`, including Joblib
and native LightGBM models, metrics, feature importance, and validation
predictions.

### Water proximity features

Both the neural and LightGBM pipelines calculate `distance_to_water_m` from a
property coordinate to the nearest retained OSM water boundary, retaining it
for diagnostics and map display. The model-facing continuous feature is
`water_proximity = exp(-distance_to_water_m / 100)`, which is about `0.0067` at
500 metres. `is_waterfront` remains `1` at 50 metres or less and `0` otherwise.
Lakes, river-area boundaries, canal-area boundaries, and coastline are
included; `waterway=*` centerlines are not used.

The canonical geometry is `data/osm/king_county_water.geojson`. Rebuild it from
the local Washington PBF with:

```bash
python3 scripts/extract_osm_water.py
```

To retrain both models, export their ONNX artifacts, and copy everything needed
by the Java app:

```bash
make retrain-deploy
```
