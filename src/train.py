import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import Config
from model import LogisticRegressionModel

from loguru import logger

class Train:
    """Training pipeline."""

    def __init__(self):
        self.CONF = Config()

        self.lr_model = LogisticRegressionModel()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load preprocess data from csv files"""
        logger.info("Loading input csv files.")
        self.df = pd.read_csv(
            self.CONF.preprocess_config.PATH_LOCAL_PREPROCESSED_DATA
        )
        logger.info("Dataset loaded.")

    def train_test_split(self) -> None:
        """Split training data into train and validation datasets
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop(self.CONF.preprocess_config.TARGET, axis=1),
            self.df[self.CONF.preprocess_config.TARGET],
            train_size=0.8, 
            test_size=0.2, 
            stratify=self.df[self.CONF.preprocess_config.TARGET]
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit_model(self):
        """Fit model onto training dataset."""
        logger.info("Fitting model...")
        self.lr_model.fit(self.X_train, self.y_train)
        logger.info("Model fitted.")

    def evaluate_model(self) -> None:
        """Evaluate the model on validation data."""
        train_predictions = self.lr_model.predict(self.X_train)
        valid_predictions = self.lr_model.predict(self.X_test)

        logger.info("Classification matrix on validation dataset")
        logger.info(confusion_matrix(self.y_test, valid_predictions))

        logger.info(f"Training roc auc score {roc_auc_score(self.y_train, train_predictions)}")
        logger.info(f"Training roc auc score {roc_auc_score(self.y_test, valid_predictions)}")

    def main(self) -> None:
        """Training pipeline."""
        self.load_data()
        self.train_test_split()
        self.fit_model()
        self.evaluate_model()