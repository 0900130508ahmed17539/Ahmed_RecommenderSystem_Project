# Foundational Matrix Factorization for Recommender Systems

**Ahmed Mohammad Ahmed**  
**Student No: 29799899**  
**Stellenbosch University**  
📧 29799899@sun.ac.za


## Overview

In our code we build a recommender system from scratch using alternating least squares (ALS) for the MovieLens 32 million dataset. Our system solves (address) sparse matrix completion with 200,948 users, 84,432 movies, and only 0.19% of possible ratings observed.

In any system selling products, popular products occupy one end while a long tail of content remains a heavy tail of products users do not know about but that remains monetizable. To capture this value, we build multiple ALS models starting from simple biases to complex uncertainty quantification. Each model addresses different aspects of the recommendation problem.

We hope this creates value for both users discovering content and businesses monetizing their entire catalog.

## What We Achieved

* **0.7716 RMSE** on test data through systematic hyperparameter optimization
* **0.7708 RMSE** with hierarchical features for cold start problems
* **0.8460 RMSE** with variational inference providing uncertainty quantification
* **BPR implementation** achieving 0.9953 AUC for implicit feedback
* **A/B testing framework** showing 32.7% improvement (p < 0.05)

### Model Performance Comparison

| Model | Train RMSE | Test RMSE | Key Feature |
|-------|------------|-----------|-------------|
| Bias-Only | 0.8462 | 0.8558 | Simple baseline |
| Latent ALS | 0.6721 | 0.7716 | Latent model |
| Hierarchical | 0.6854 | 0.7708 | For cold start |
| Variational | 0.8389 | 0.8460 | For uncertainty estimates |

## Requirements

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## Dataset Setup

1. Download MovieLens-32M dataset from [GroupLens](https://grouplens.org/datasets/movielens/32m/)
2. Extract `ratings.csv` and `movies.csv` to your project folder
3. Update file paths in the code:
```python
ratingspath = "/path/to/your/ratings.csv"
moviespath = "/path/to/your/movies.csv"
```

## Small Tips on How to Run Our Code

Welcome! To run our system:

1. Update the data paths to your MovieLens CSV files in the main function () the last one in our file.
2. Run the Python code cell

That's it! Our code will trains all models and generates visualizations.

### What Happens When We Run

1. **Data Loading**: We load 32 million ratings and create sparse matrix structures
2. **Baseline Model**: We start with bias-only model to test infrastructure
3. **Hyperparameter Tuning**: We test λ ∈ [0.1, 0.2], τ ∈ [1.0, 2.0], γ ∈ [2.5, 5.0]
4. **ALS Training**: We train latent factor model with D=20 dimensions
5. **Hierarchical Features**: We incorporate genre information for cold start
6. **BPR Training**: We handle implicit feedback scenarios
7. **Variational Inference**: We add uncertainty quantification
8. **A/B Testing**: We simulate production experiments

## Code Organization

To start We organized our code into separate classes to make it clean and easy to work with. Each model gets its own class, which means: which help use us for instance If our BPR model has a bug, we know exactly where to look - in the BPR class. We don't need to lose our mind through thousands of lines of mixed code.

## Our Model Classes

- **`DataLoader`**: For loading our 32 million ratings and creates the data structures we need to train our models (handles sparse matrix with 99.81% sparsity)
- **`BiasOnly`**: Our simplest baseline using just user/item biases (0.8558 RMSE)
- **`Latent`**: Our main ALS model with embeddings for personalization (0.7716 RMSE)
- **`Features`**: Our solution for cold-start movies with genres (0.7708 RMSE)  
- **`VI`**: Our model that provides uncertainty estimates (0.8460 RMSE)
- **`BPR`**: Our approach for implicit feedback when users don't rate (0.9953 AUC)
- **`ABTest`**: Our production testing framework using AB Testing (32.7% improvement)

## Memory Requirements

* Minimum: 8GB RAM
* Recommended: 16GB RAM
* VI model requires 2x memory for storing variances

## GitHub

https://github.com/0900130508ahmed17539

## Author

Ahmed Mohammad Ahmed -  Stellenbosch University . African Institute for Mathematical Sciences (AIMS) South Africa,
