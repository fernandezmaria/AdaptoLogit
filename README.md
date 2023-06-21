# `AdaptoLogit` package

## Introduction
`AdaptoLogit` is a python package that proposes the implementation of adaptive lasso to solve logistic regression models. 

## Dependencies
`AdaptoLogit` requires:
- Numpy >= 1.2
- SciPy >= 1.7.0
- Scikit-learn >= 1.0

## User installation
If you already have a working installation of numpy, scipy and scikit-learn, the easiest way to install **AdaptoLogit** is using pip:

```sh
pip install AdaptoLogit
```

## Usage Example
In the following example, the package is used to apply adaptive lasso logistic regression on simulated binary data. Cross validation is carried out to get the optimal subset of parameters for the data. 
```py
from sklearn.model_selection import GridSearchCV
import numpy as np
import AdaptoLogit as al
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(random_state=8) # 100 samples, 20 features, 2 informative

# Estimate weights
weight = al.AdaptiveWeights(power_weight=(0.5,1,1.5))
weight.fit(X, y)

# Build model 
model = al.AdaptiveLogistic()

# Cross validation for best subset of parameters
param_grid = {'C':[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'weight_array': weight.lasso_weights_,
            'solver': ['saga','liblinear'], 'max_iter': [1000, 10000]}     
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
grid_search.fit(X, y)
final_model = grid_search.best_estimator_

# Model coefficients
print("Model coefficients ", final_model.coef_)
```
