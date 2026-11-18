# MovieLens Recommender: ALS and Neural Collaborative Filtering

## Overview

This project compares Alternating Least Squares (ALS), using the `implicit` library, with Neural Collaborative Filtering (NCF), implemented in PyTorch, on the MovieLens-25M dataset. It uses a chronological train/validation/test split and evaluates recommendation quality with ranking metrics including NDCG@10.

## Data Pipeline

The notebook loads MovieLens ratings, movies, and tags, preprocesses the data, creates a temporal split, and writes intermediate datasets as Parquet files. It also explores rating and popularity distributions and builds user/item indices for model training. The notebook currently contains local machine-specific paths; update them before running elsewhere.

## Models and Results

- **ALS:** Hyperparameter search over latent factors, regularization, and confidence scaling. The reported unbiased test NDCG@10 is 0.0290 and Precision@10 is 0.0495.
- **NCF:** PyTorch model with user/item embeddings and an MLP. The reported test RMSE is 0.8119.
- A popularity baseline is included for cold-start users.

These are the results currently documented in the project; the ALS ranking metric and NCF rating-prediction metric are different measures and should not be treated as a direct head-to-head comparison.

## 🚀 How to Run

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/Ahmad-lababidi/movielens-recommender-als-ncf.git
   cd movielens-recommender-als-ncf
   pip install -r requirements.txt
   ```
2. Download [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/) and extract the dataset into an `ml-25m/` folder. Update the notebook's `PATH`, `OUT_PATH`, and `FIGURES_PATH` values to point to valid local folders.
3. Open [`movielens-recommendation-als-ncf.ipynb`](movielens-recommendation-als-ncf.ipynb) and run the cells. NCF training with automatic mixed precision requires a CUDA-capable GPU; CPU execution may require adjusting the notebook configuration.
