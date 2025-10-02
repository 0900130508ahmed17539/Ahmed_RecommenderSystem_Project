# Foundational Models for Recommender Systems

## Overview
IWe build a complete recommender system from scratch using alternating least squares (ALS) for the MovieLens 32 million dataset. Our system solves sparse matrix completion with 200,948 users, 84,432 movies, and only 0.19% of possible ratings observed.

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
| Latent ALS | 0.6721 | 0.7716 | Best accuracy |
| Hierarchical | 0.6854 | 0.7708 | Cold start handling |
| Variational | 0.8389 | 0.8460 | Uncertainty estimates |

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

## How to Run

### Quick Start
```bash
python als_recommender.py
```

### What Happens When You Run
1. **Data Loading**: We load 32 million ratings and create sparse matrix structures
2. **Baseline Model**: We start with bias-only model to test infrastructure
3. **Hyperparameter Tuning**: We test λ ∈ [0.1, 0.2], τ ∈ [1.0, 2.0], γ ∈ [2.5, 5.0]
4. **ALS Training**: We train latent factor model with D=20 dimensions
5. **Hierarchical Features**: We incorporate genre information for cold start
6. **BPR Training**: We handle implicit feedback scenarios
7. **Variational Inference**: We add uncertainty quantification
8. **A/B Testing**: We simulate production experiments

## Our Architecture

### Why Object-Oriented Programming
We chose object-oriented programming to create a modular system where each component has clear responsibilities. This design allows us to:
- Isolate different algorithms for easy testing and comparison
- Share common interfaces between models (train, predict, evaluate)
- Maintain clean separation between data handling, model training, and evaluation
- Enable easy extension with new models without modifying our existing code

### Core Classes
```python
DataLoader()       # Handles sparse matrix with 99.81% sparsity
BiasOnly()        # Baseline with user/item biases
Latent()          # Matrix factorization with ALS
Features()        # Hierarchical model with genres
BPR()             # Implicit feedback with ranking
VI()              # Uncertainty quantification
ABTest()          # Production testing using AB Testing 
```

Each class encapsulates its own state and behavior. For instance, our Latent class maintains user matrices U and item matrices V, while our VI class extends this to also store variance parameters. This modular design means we can swap models easily - all models expose the same train() and computermse() methods.

## Memory Requirements
* Minimum: 8GB RAM
* Recommended: 16GB RAM
* VI model requires 2x memory for storing variances

## Customization

### Adjust Hyperparameters
```python
model.train(D=20, epochs=10, lam=0.1, gam=2.5, tau=1.0)
```

### Test Specific Models Only
Comment out sections in main to run specific components:
```python
# Skip to test only BPR
# biasmodel.train()  # Comment this
# alsmodel.train()   # Comment this
bpr.fit()            # Run only this
```

## Files
* `als_recommender.py` - Complete implementation
* `ratings.csv` - MovieLens ratings (download separately)
* `movies.csv` - Movie metadata with genres (download separately)

## GitHub
https://github.com/0900130508ahmed17539

## Author
Ahmed Mohammad - African Institute for Mathematical Sciences (AIMS) South Africa, Stellenbosch University
