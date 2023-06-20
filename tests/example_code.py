import AdaptoLogit as al

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from numpy.testing import assert_allclose
from sklearn.model_selection import GridSearchCV
import numpy as np

# Generate data
X, y = make_classification() #100 samples, 20 features, 2 informative

# Compare against logistic regression
logistic_model = LogisticRegression(C=100, penalty='l1', solver='liblinear', random_state=100)
logistic_model.fit(X, y)

al_model = al.AdaptiveLogistic(C=100, random_state=100) # none weight array
al_model.fit(X, y)

assert_allclose(logistic_model.coef_, al_model.coef_)  # No output means OK

# Estimate weights
weight = al.AdaptiveWeights(weight_technique="svc", C=[1, 10, 100, 1000],gamma=[1, 0.1, 0.001, 0.0001],
                            kernel=['linear'])
weight.fit(X, y)

# #debug
# print("- Instance of the class")
# print(', '.join("%s: %s" % item for item in vars(weight).items()))

# Build model
model = al.AdaptiveLogistic(C=100, random_state=100, weight_array=weight.lasso_weights_[0])
model.fit(X, y)
print("- Adaptive Lasso coefficients: ", model.coef_)

# Use Cross validation
X, y = make_classification(n_samples=1000, n_features=100, random_state=100)

weight = al.AdaptiveWeights(power_weight=(0, 0.5, 1, 1.5))
weight.fit(X, y)

model = al.AdaptiveLogistic()
param_grid = {'C':[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'weight_array': weight.lasso_weights_}


grid_search = GridSearchCV(model,
                           param_grid,
                           cv=3,
                           scoring='accuracy',
                           n_jobs=11)
grid_search.fit(X, y)

final_model = grid_search.best_estimator_
w = final_model.get_params()['weight_array']

# Check for which power_weight is found the optimal solution
for i, arr in enumerate(weight.lasso_weights_):
    if np.array_equal(arr, w):
        print(f"Target array found at index {i}")
        break
else:
    print("Target array not found in list")