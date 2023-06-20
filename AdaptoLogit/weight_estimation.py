from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


class AdaptiveWeights(BaseEstimator):
    """
    Base class for Adaptive Lasso logistic regression that includes the code to compute the weights of each of the input features.

    Parameters
    ----------
    weight_technique: {'pca_1', 'pca_pct', 'spca', 'ridge', 'svc', 'xgb'}, default='pca_pct'.
        Specify the weight calculation method

    power_weight: int or list of floats, default = (1, 1.5)
        Each of the values describes the strength of the coefficients when calculating the weights.

    weight_tol: float, default=1e-4
        tolerance for the coefficients to be taken as 0.

    variability_pct: float, default=0.9
        It must be between 0 and 1 as it indicates the percentage of variability explained to be considered in the
        'pca_pct' and 'spca' techniques

    extra_parameters: additional argument that allows the user to enter specific parameters that can be used in the
        'svc', 'xgb' and 'ridge' methods.
        All parameters entered will be collected as a Python dictionary and will be entered as input in the
        hyper-tuning of parameters of these methods.

    Attributes
    ----------
    lasso_weights_: ndarray of shape (number of power weights, number of features)
        N-dimensional matrix including the weight of each feature to be applied to the penalty term in the
        'AdaptiveLogistic' class. It contains the weights for each of the values of the 'power_weight' parameter.

    n_features_in_ : int
        Number of features seen during term:'fit'.

    Example
    -------
    >> import AdaptoLogit as al
    >> from sklearn.datasets import make_classification

    >> X, y = make_classification()
    >> weight = al.AdaptiveWeights(weight_technique="svc", C=[1, 10, 100, 1000],gamma=[1, 0.1, 0.001, 0.0001],
                                   kernel=['linear'])
    >> weight.fit(X, y)
    >> weight.lasso_weights_
    [array([ -2.2779018 ,   2.13228948,   1.85153   ,   7.15353933,
              0.47237009,   1.6884743 ,  -1.05206186,  -2.55093836,
             -2.10438012,  -3.23987963, -14.87976718,  -5.70170631,
             -1.78714711,   1.08736836,   1.23749567,   1.91785501,
             49.50017907,   4.316734  ,  10.62704557,  23.70737323]), array([-3.43798047e+00,  3.11367243e+00,
            2.51945917e+00,  1.91168619e+01,  3.24668581e-01,  2.19409984e+00, -1.07904671e+00, -4.07437220e+00,
           -3.05268341e+00, -5.83224159e+00, -5.75996626e+01, -1.36215683e+01,
           -2.38906400e+00,  1.13393100e+00,  1.37669064e+00,  2.65603163e+00,
            3.38973754e+02,  8.96653435e+00,  3.45785790e+01,  1.14516822e+02])]
    """

    def __init__(self, weight_technique='pca_pct', power_weight=(1, 1.5), weight_tol=1e-4, variability_pct=0.9,
                 **extra_parameters):
        """
        Class definition as stated before
        """
        self.weight_technique = weight_technique
        self.power_weight = power_weight
        self.weight_tol = weight_tol
        self.variability_pct = variability_pct

        self.extra_parameters = extra_parameters

    def _pca_1(self, X, y):
        """
        Computes the adaptive weights based on the first principal component
        """
        pca = PCA(n_components=1)
        pca.fit(X)
        tmp_weight = np.abs(pca.components_).flatten()

        return tmp_weight

    def _pca_pct(self, X, y):
        """
        Computes the adpative weights based on a subset of components.
        To carry out the process of obtaining the principal components along with the application of an unpenalised
        regression model, a pipeline has been implemented that executes these steps in sequence and in order.
        """
        pipePCA = Pipeline(steps=[('pca', PCA(n_components=self.variability_pct)),
                                  ('classifier', LogisticRegression(penalty='none'))])
        pipePCA.fit(X, y)
        coef = pipePCA.named_steps.classifier.coef_
        components = pipePCA.named_steps.pca.components_
        tmp_weight = np.dot(coef, components)
        tmp_weight = np.abs(tmp_weight)

        return tmp_weight

    def _spca(self, X, y):
        """
        Computes the adpative weights based on a subset of sparse pca components.
        Same process as before.
        """
        pipeSPCA = Pipeline(steps=[('spca', PCA(n_components=self.variability_pct)),
                                   ('classifier', LogisticRegression(penalty='none'))])
        pipeSPCA.fit(X, y)
        coef = pipeSPCA.named_steps.classifier.coef_
        components = pipeSPCA.named_steps.spca.components_
        tmp_weight = np.dot(coef, components)
        tmp_weight = np.abs(tmp_weight)

        return tmp_weight

    def _svc(self, X, y):
        """
        Computes the adpative weights based on support vector machines' coefficients
        In order to obtain a better performance of the algorithm, an optimisation
        of the hyperparameters with cross-validation is carried out. If the grid of
        hyperparameters is not received as input, the values fixed in the function are used.
        """
        svc = svm.SVC()
        if self.extra_parameters.__len__() == 0:
            param_grid = {'C': [1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.001, 0.0001],
                          'kernel': ['linear']}  # coefs only available when linear #sí converge

        else:
            param_grid = self.extra_parameters

        GS_svc = GridSearchCV(svc, param_grid, cv=3, verbose=0, scoring='accuracy')
        GS_svc.fit(X, y)
        model_svc = GS_svc.best_estimator_
        tmp_weight = np.abs(model_svc.coef_).flatten()

        return tmp_weight

    def _xgb(self, X, y):
        """
        Computes the adpative weights based on xgboost feature importances.
        In the same way as in the previous case, the XGBoost is executed by means
        of cross-validation for hyperparameter tuning. It shall be carried out
        with a fixed parameter grid unless another specific combination of values
        is entered as input.
        """

        xgb = XGBClassifier()

        if self.extra_parameters.__len__() != 0:
            params = self.extra_parameters
        else:
            params = {
                'learning_rate': [0.1, 0.01, 0.001],  # eta in r
                'max_depth': [2, 4, 6, 8, 10],
                'min_child_weight': [5, 10, 15],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.2, 0.4, 0.8],
                'gamma': [0, 1],
                'objective': ['binary:logistic']
            }  # sí converge

        GS_xgb = GridSearchCV(xgb, param_grid=params, cv=3, scoring="accuracy", n_jobs=4, verbose=0)
        GS_xgb.fit(X, y)
        model_xgb = GS_xgb.best_estimator_
        tmp_weight = np.abs(model_xgb.feature_importances_)

        return tmp_weight

    def _ridge(self, X, y):
        """
        Computes the adaptive weights based on ridge coefficients.
        The procedure followed is similar to the previous two.
        """
        ridge = LogisticRegression()
        if self.extra_parameters.__len__() != 0:
            parameters = self.extra_parameters
        else:
            parameters = {'penalty': ['l2'],
                          'C': [0.1, 1, 10, 100],
                          'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                          'max_iter': [100, 300, 500]
                          }  # sí converge
        GS_ridge = GridSearchCV(ridge, parameters, cv=3, verbose=0, scoring='accuracy')
        GS_ridge.fit(X, y)
        model_ridge = GS_ridge.best_estimator_
        tmp_weight = np.abs(model_ridge.coef_).flatten()

        return tmp_weight

    def fit(self, X, y=None):
        """
        This is the main function of the class.
        Computes the weights according the parameters specified in the class definition

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (number of samples, number of features)
            Training vector for the different methods.
        y : array-like of shape (number of samples, number of classes)
           Target vector relative to X. In this case the class is binary.

        Returns
        -------
        self: fitted weights.
        """
        if y is not None:
            X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        tmp_weight = getattr(self, '_' + self.weight_technique)(X=X, y=y)
        if np.isscalar(self.power_weight):
            self.lasso_weights_ = [1 / (tmp_weight ** self.power_weight + self.weight_tol)]
        else:
            self.lasso_weights_ = [1 / (tmp_weight ** elt + self.weight_tol) for elt in self.power_weight]
        self.is_fitted_ = True
        return self