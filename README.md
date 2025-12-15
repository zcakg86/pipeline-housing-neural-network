# Project: Transformer-Based Real Estate Price Prediction

### **Executive Summary**
This project implements an end-to-end deep learning pipeline designed to forecast real estate prices. It utilizes a custom **Attention-Based Neural Network** that treats tabular property data as a sequence of tokens, allowing the model to learn complex, non-linear relationships between location, time, and physical property characteristics.

# Aims
* Implement location-specific embedding through spatial features and community detection, that can estimate prices without bias across King County.
* Provide accurate estimates throughout time.
* Produce dashboard map to present model predictions and performance.

# Methods
* H3 Index for spatial representation
* Louvain graph based community detection
* Temporal embedding layer

# Data
Residential Sales data:\
Andy Krause https://www.kaggle.com/datasets/andykrause/kingcountysales/data Version 8: kingco_sales.csv\
Data produced from property assessment data made available by the King County Department of Assessments.\
See https://github.com/andykrause/kingCoData

### **Neural Network Architecture**
The core model utilizes a Transformer-inspired architecture adapted for structured tabular data:

*   **Input Representation:** The model ingests heterogeneous data types by projecting them into a shared latent embedding space:
*   **Categorical Embeddings:** Learnable vector representations for *Community* (Location), *Year*, and *Week* (Seasonality).
*   **Numerical Projections:** Linear transformation of physical features (*Sqft, Lot Size, Beds*).
*   **Sequence Construction:** These feature vectors are stacked to form a sequence, prepended by a learnable **`[CLS]` (Classification) Token**.
*   **Self-Attention Mechanism:** A Multi-Head Attention layer enables every feature to contextually interact with every other feature. The `[CLS]` token aggregates a global summary of the property by "attending" to the specific nuances of the location, time, and physical specs.
*   **Regression Head:** The final price prediction is generated via a Multi-Layer Perceptron (MLP) that processes only the contextualized `[CLS]` output, effectively using it as a learned embedding of the entire listing.

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
