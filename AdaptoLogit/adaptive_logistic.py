from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y
import numpy as np


class AdaptiveLogistic(LogisticRegression):
    """
    Adaptive Lasso Logistic Regression classifier.

    The class implements adaptive lasso logistic regression for binary data. The weights should be previously computed and introduced from the
    'AdaptiveWeights' class.

    Parameters
    ----------
    penalty :'l1', default = 'l1'
        In order to apply the lasso penalty, this parameter is set to 'l1'.

    weight_array: one-dimensional array of shape (1, number of features), default=None
        Vector of weights to apply in the penalty term.

    C: float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    solver: {'liblinear', 'saga'}, default='liblinear'
        Algorithm to use in the optimization problem. The solvers supported for this regularization are 'liblinear' and 'saga'.
        To choose one, notice the following aspects:
            - For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones;

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    Attributes
    ----------
    coef_ : one-dimensional array of shape (1, number of features)
        Coefficient of the features in the decision function.

        This is the array of coefficients already transformed, since in order to construct them for the adaptive lasso it is
        necessary to divide the coefficents obtained by the decision function by the weights.

    Notes
    -----
    Given that the LogisticRegression class of the scikit-learn package is taken as a base, the rest of the parameters and attributes
    contained in this class are also inherited.

    References
    ----------
    Scikit-learn -- Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

    Examples
    --------
    >> import AdaptoLogit as al
    >> from sklearn.datasets import make_classification

    >> # Generate data
    >> X, y = make_classification() # 100 samples, 20 features, 2 informative

    >> al_model = al.AdaptiveLogistic(C=100, random_state=100) # none weight array
    >> al_model.fit(X, y)
    >> al_model.coef_
    array([[ 0.        , -0.02877044, -0.03607249,  0.39449766, -0.36033761,
             0.35230243, -0.37762612,  0.27196336,  0.83807621, -0.07779732,
             0.16500129,  3.36619351,  0.06285085,  0.02813974,  0.        ,
             0.34837082,  0.1788243 ,  0.23302661, -0.39078461,  0.19451076]])

    Using cross validation from scikit-learn
    ----------
    >> from sklearn.model_selection import GridSearchCV
    >> import numpy as np

    >> # Generate data
    >> X, y = make_classification(random_state=8) # 100 samples, 20 features, 2 informative

    >> # Estimate weights
    >> weight = al.AdaptiveWeights(power_weight=(0.5,1,1.5))
    >> weight.fit(X, y)

    >> # Build model
    >> model = al.AdaptiveLogistic()

    >> # Cross validation for best subset of parameters
    >> param_grid = {'C':[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'weight_array': weight.lasso_weights_,
                    'solver': ['saga','liblinear'], 'max_iter': [1000, 10000]}
    >> grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
    >> grid_search.fit(X, y)
    >> final_model = grid_search.best_estimator_

    >> # Check for which power_weight is found the optimal solution
    >> w = final_model.get_params()['weight_array']
    >> for i, arr in enumerate(weight.lasso_weights_):
           if np.array_equal(arr, w):
                print(f"Optimal power_weight value is {weight.power_weight[i]}")
                break
    >> else:
            print("Target array not found in list")
    Optimal power_weight value is 1

    >> # Model coefficients
    >> print("Model coefficients ", final_model.coef_)
    Model coefficients  [[ 0.          0.          0.          0.          0.          0.
    0.97572775  0.          0.          0.          0.          0.           0.          0.
    0.          0.          0.         -0.31962469         0.          0.        ]]
    """

    def __init__(self, weight_array=None, C=1.0, penalty='l1', solver='liblinear', max_iter=100, **kwargs):
        """
        Class definition
        """
        self.weight_array = weight_array
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        super().__init__(C=C, penalty=penalty, solver=solver, max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (number of samples, number of features)
            Training vector.
        y : array-like of shape (number of samples, number of classes)
            Target vector relative to X. In this case the class is binary.

        Returns
        -------
        self
            Fitted estimator.
        """
        if y is not None:
            X, y = check_X_y(X, y)
        if self.weight_array is None:
            self.weight_array = np.ones(X.shape[1])
        X_weighted = X / self.weight_array
        super().fit(X_weighted, y)
        self.is_fitted_ = True
        # Model coefficients transformed
        self.coef_ = self.coef_ / self.weight_array
        return self