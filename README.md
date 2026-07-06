# Project: Transformer-Based Real Estate Price Prediction

### **Executive Summary**
This project implements an end-to-end deep learning pipeline designed to forecast real estate prices. It utilizes a custom **Attention-Based Neural Network** that treats tabular property data as a sequence of tokens, allowing the model to learn complex, non-linear relationships between location, time, and physical property characteristics.

### Aims
* Implement location-specific embedding through spatial features and community detection, that can estimate prices without bias across King County.
* Provide accurate estimates throughout time.
* Produce dashboard map to present model predictions and performance.


# Data
Residential Sales data:\
Andy Krause https://www.kaggle.com/datasets/andykrause/kingcountysales/data Version 8: kingco_sales.csv\
Data produced from property assessment data made available by the King County Department of Assessments.\
See https://github.com/andykrause/kingCoData

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
*   **Java Export:** Model export is supported for Java deployment via `export_model_for_java.py`.

### **Java Integration**
See [java-app/house-price-app/README.md](https://github.com/zcakg86/pipeline-housing-neural-network/tree/kiraze/java-app/house-price-app) for the Java app and how the exported model is consumed.
