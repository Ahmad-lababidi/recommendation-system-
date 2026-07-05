# recommendation-system-

# Movie Recommendation System: Collaborative Filtering & Deep Learning Approaches

## Authors

* **Ahmad Lababidi**

## 📌 Project Overview

This project implements and compares two advanced recommendation system architectures—Alternating Least Squares (ALS) and Neural Collaborative Filtering (NCF)—using the MovieLens-25M dataset. The objective is to build a robust engine capable of generating personalized top-K movie recommendations while addressing common industry challenges such as cold starts, data sparsity, and model overfitting.

## 📊 Data Pipeline & Exploratory Data Analysis

The system strictly enforces data integrity and optimizes memory by utilizing `int32` and `float32` datatypes during ingestion. The raw dataset undergoes comprehensive preprocessing:

* **Sparsity & Cold Start Mitigation:** The dataset is filtered to exclude users and movies with insufficient interaction history (a minimum threshold of 20 interactions was applied for deep learning training, yielding a highly concentrated matrix).


* **Temporal Splitting:** To prevent data leakage and simulate real-world deployment, the dataset is split chronologically. The final 30 days of interactions serve as the unseen Test set, and the prior 30 days act as the Validation set. Ratings below 3.5 are dropped from validation and test splits to focus evaluation exclusively on positive recommendations.


* **Storage Optimization:** Cleaned data splits are exported as columnar `.parquet` files for significantly faster read/write operations.


* **EDA Insights:** Exploratory visualizations were generated to analyze the distribution of ratings, the long-tail distributions of both user activity and movie popularity, and the frequency of movie genres.



## 🧠 Modeling Architectures

### 1. Matrix Factorization: Alternating Least Squares (ALS)

Built using the `implicit` library, this model relies on collaborative filtering via matrix factorization.

* **Hyperparameter Search:** A grid search was conducted over latent factors (32, 64, 128), regularization constraints (0.01, 0.05), and confidence scaling alpha variables (15, 40, 80).


* **Optimal Configuration:** The highest Validation NDCG@10 was achieved with 64 factors, a regularization penalty of 0.01, and an alpha of 15.



### 2. Neural Collaborative Filtering (NCF)

A deep learning approach built with PyTorch, designed to capture complex, non-linear user-item interactions.

* **Architecture:** The network initializes 64-dimensional latent embedding layers for users and items. These vectors are concatenated and passed through a Multi-Layer Perceptron (MLP) with a `[128, 64, 32]` structure.


* **Overfitting Prevention:** L2 Regularization (Weight Decay of 1e-4) and a 0.3 Dropout rate are applied within the MLP.


* **Training Optimization:** The model utilizes the AdamW optimizer, Mean Squared Error (MSE) loss, and PyTorch's Automatic Mixed Precision (AMP) `GradScaler` for accelerated GPU computation. Early stopping is configured with a patience of 2 epochs to restore optimal weights once validation loss plateaus.



## 📈 Evaluation & Key Results

The models were evaluated against strict, unseen temporal test sets to verify real-world predictive power.

* **ALS Performance:** Achieved an Unbiased Test NDCG@10 of **0.0290** and a Precision@10 of **0.0495**.


* **NCF Performance:** Achieved a Final Test RMSE of **0.8119**.


* **Behavioral Metrics (NCF):**
* **Catalog Coverage:** Evaluated on a 500-user sample, the model successfully recommended 332 unique items, yielding a coverage of **1.80%**.


* **Intra-list Diversity:** By computing pairwise cosine similarity of latent item embeddings, the model achieved a diversity score of **0.3991**.




* **Baseline Fallback:** A popularity-based baseline is automatically triggered for new "Cold Start" users lacking historical data.



## ⚙️ System Requirements & Execution

**Hardware Note:** To leverage the Automatic Mixed Precision (AMP) during the NCF training phase, a CUDA-compatible GPU is required. The models were successfully trained and validated on a system utilizing 32GB of RAM and an NVIDIA RTX 3050 Ti (4GB VRAM).

**Execution Assumptions & Common Mistakes:**

* **MKL Threading:** When training the ALS model on a CPU, ensure the environment variable `MKL_NUM_THREADS=1` is set to prevent internal threadpool conflicts and performance degradation.


* **DataLoaders:** For PyTorch `DataLoader` stability, particularly in Windows environments, `num_workers` is intentionally set to `0` to prevent multiprocessing crashes.
