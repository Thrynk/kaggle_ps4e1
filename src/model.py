from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(LogisticRegression):
    """Logistic Regression model from scikit-learn api.
    """

    def __init__(self) -> None:
        super(LogisticRegressionModel, self).__init__(class_weight='balanced')

    def main(self) -> None:
        return self
    
    