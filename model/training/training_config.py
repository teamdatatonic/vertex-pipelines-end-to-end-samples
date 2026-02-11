from pydantic import BaseModel


class TrainingConfig(BaseModel):

    label: str
    n_estimators: int
    early_stopping_rounds: int
    objective: str
    booster: str
    learning_rate: float
    min_split_loss: float
    max_depth: int
    train_test_split_size: float
    train_valid_split_size: float
    train_test_random_state: int
    train_valid_random_state: int

    def get_model_params(self) -> dict:
        """Extract only XGBoost model parameters."""
        return {
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "objective": self.objective,
            "booster": self.booster,
            "learning_rate": self.learning_rate,
            "min_split_loss": self.min_split_loss,
            "max_depth": self.max_depth,
        }

    def get_split_params(self) -> dict:
        """Extract train/test split parameters."""
        return {
            "train_test_split": {
                "test_size": self.train_test_split_size,
                "random_state": self.train_test_random_state,
            },
            "train_valid_split": {
                "validation_size": self.train_valid_split_size,
                "random_state": self.train_valid_random_state,
            },
        }


# TODO: Add preprocessing configuration
# Future enhancement: Make feature columns and preprocessing strategies configurable
# def get_preprocess_params(self) -> dict:
