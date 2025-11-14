"""
Basic model training and evaluation examples.
"""

from models.training import ols_basic
from models.evaluation import no_scaling_logistic_prediction, no_scaling_linear_prediction


def ols():
    """
    Run basic OLS model training.
    """
    ols_basic()


def main():
    """
    Main function to run the basic model.
    """
    ols()


if __name__ == '__main__':
    main()