import unittest
import AdaptoLogit as al
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from numpy.testing import assert_allclose


class MyTestCase(unittest.TestCase):
    def test_lasso(self):
        """
        Test that the AdaptiveLogistic with no weights gives the same result as LogisticRegression
        :return:
        """
        X, y = make_classification()
        # Logistic regression model
        logistic_model = LogisticRegression(C=100, penalty='l1', solver='liblinear', random_state=100)
        logistic_model.fit(X, y)

        # Adaptive logistic
        al_model = al.AdaptiveLogistic(C=100, random_state=100)
        al_model.fit(X, y)
        assert_allclose(logistic_model.coef_, al_model.coef_)


if __name__ == '__main__':
    unittest.main()
