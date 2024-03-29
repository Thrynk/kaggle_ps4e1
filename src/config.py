from typing import List
from pydantic import BaseSettings, Field

class PreprocessConfig(BaseSettings):
    PATH_LOCAL_RAW_DATA: str = Field(
        default="data/raw_data.csv",
        description="Path to local raw data csv file.",
        env="PATH_LOCAL_RAW_DATA"
    )

    PATH_LOCAL_PREPROCESSED_DATA: str = Field(
        default="data/preprocess_data.csv",
        description="Path to preprocessed data csv file.",
        env="PATH_LOCAL_PREPROCESSED_DATA"
    )

    CAT_COLUMNS: List[str] = Field(
        default=["Geography", "Gender"],
        description="List of categorical variables which needs to be preprocessed",
        env="CAT_COLUMNS"
    )

    ORDINAL_CAT_COLUMNS: List[str] = Field(
        default=["NumOfProducts", "HasCrCard", "IsActiveMember", "Tenure"],
        description="List of ordinal categorical variables which don't need to be preprocessed",
        env="ORDINAL_CAT_COLUMNS"
    )

    NUMERICAL_COLUMNS: List[str] = Field(
        default=["CreditScore", "Age", "Balance", "EstimatedSalary"],
        description="List of numerical variables which needs to be preprocessed",
        env="NUMERICAL_COLUMNS"
    )

    TARGET: str = Field(
        default="Exited",
        description="Target column name.",
        env="TARGET"
    )

    class Config:
        case_sensitive = True
        env_file = ".env"

class Config(BaseSettings):
    preprocess_config: PreprocessConfig = PreprocessConfig()