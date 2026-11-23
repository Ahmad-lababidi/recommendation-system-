# MovieLens Recommender: ALS and Neural Collaborative Filtering

## Overview

This project compares Alternating Least Squares (ALS), using the `implicit` library, with Neural Collaborative Filtering (NCF), implemented in PyTorch, on the MovieLens-25M dataset. It uses a chronological train/validation/test split and evaluates recommendation quality with ranking metrics including NDCG@10.

## Data Pipeline

The notebook loads MovieLens ratings, movies, and tags, preprocesses the data, creates a temporal split, and writes intermediate datasets as Parquet files. It also explores rating and popularity distributions and builds user/item indices for model training. The notebook currently contains local machine-specific paths; update them before running elsewhere.

## Models and Results

| Model | What it does | Test result |
|---|---|---|
| **ALS** (`implicit`) | Matrix factorization with a hyperparameter search over latent factors, regularization and confidence scaling | NDCG@10 **0.0290** · Precision@10 **0.0495** |
| **NCF** (PyTorch) | User and item embeddings fed into an MLP, trained with mixed precision | RMSE **0.8119** |
| **Popularity baseline** | Fallback recommendations for cold-start users | — |

ALS is scored on ranking and NCF on rating prediction, so the two numbers measure different things and are not a head-to-head comparison.

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Ahmad-lababidi/recommendation-system-.git
   cd recommendation-system-
   pip install -r requirements.txt
   ```
2. Download [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) and extract the dataset into an `ml-25m/` folder. Update the notebook's `PATH`, `OUT_PATH`, and `FIGURES_PATH` values to point to valid local folders.
3. Open [`movielens-recommendation-als-ncf.ipynb`](movielens-recommendation-als-ncf.ipynb) and run the cells. NCF training with automatic mixed precision requires a CUDA-capable GPU; CPU execution may require adjusting the notebook configuration.
