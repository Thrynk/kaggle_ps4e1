import pandas as pd
from sklearn.preprocessing import StandardScaler

from loguru import logger
from config import Config

class Preprocess:
    """Prepare input data for training
    """

    def __init__(self) -> None:
        """Initialize preprocessing pipeline
        """
        self.raw_data: pd.DataFrame = None
        self.data_with_dummies: pd.DataFrame = None
        self.preprocessed_data: pd.DataFrame = None
        self.CONF = Config()

    def load_data(self) -> None:
        """Load the data from local to pandas DataFrame"""
        self.raw_data = pd.read_csv(
            self.CONF.preprocess_config.PATH_LOCAL_RAW_DATA
        )
        logger.info("Dataset loaded.")

    def preprocess_categorical_variables(self) -> None:
        """Preprocess categorical variables into dummy variables
        """
        self.data_with_dummies = pd.get_dummies(self.raw_data, columns=self.CONF.preprocess_config.CAT_COLUMNS)
        logger.info("Categorical features preprocessed and replaced with dummies.")

    def preprocess_numerical_variables(self) -> None:
        """Preprocess numerical variables with scaling
        """
        self.preprocessed_data = self.data_with_dummies.copy()

        std_scaler = StandardScaler()
        self.preprocessed_data.loc[:, self.CONF.preprocess_config.NUMERICAL_COLUMNS] = std_scaler.fit_transform(self.preprocessed_data[self.CONF.preprocess_config.NUMERICAL_COLUMNS])
        logger.info("Numerical features preprocessed with scaling.")

    def save_data(self) -> None:
        """Save preprocessed data.
        """
        self.preprocessed_data.to_csv(self.CONF.preprocess_config.PATH_LOCAL_TRAIN_PREPROCESSED_DATA)
        logger.info("Preprocessed data saved.")

    def main(self) -> None:
        """Main function for the preprocessing pipeline."""
        logger.info("Preprocessing pipeline started.")
        self.load_data()
        self.preprocess_categorical_variables()
        self.preprocess_numerical_variables()
        self.save_data()
        logger.info("Preprocessing pipeline finished.")