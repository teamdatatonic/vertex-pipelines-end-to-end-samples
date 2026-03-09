from pydantic import BaseModel, field_validator
from typing import Any


class PreprocessingStep(BaseModel):
    encoder: str
    columns: list[str]
    kwargs: dict[str, Any] = {}
    per_column: bool = False


_MODEL_REGISTRY: dict[str, str] = {
    "XGBRegressor": "xgboost.XGBRegressor",
}


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
    preprocessing: list[PreprocessingStep]
    model: str
    primary_metric: str

    @field_validator("model")
    @classmethod
    def model_must_be_registered(cls, v: str) -> str:
        if v not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{v}'. Must be one of: {list(_MODEL_REGISTRY)}"
            )
        return v

    def get_model_class(self) -> type:
        """Resolve the model name to its class at runtime (lazy import).

        xgboost is only imported here so the config package itself does not
        need xgboost as a dependency.
        """
        import importlib

        module_path, class_name = _MODEL_REGISTRY[self.model].rsplit(".", 1)
        return getattr(importlib.import_module(module_path), class_name)

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
                "test_size": self.train_valid_split_size,
                "random_state": self.train_valid_random_state,
            },
        }

    def get_transformers(self, X_train) -> list[tuple]:
        """Build ColumnTransformer tuples driven entirely by the preprocessing config.

        sklearn is imported lazily here so this package has no sklearn dependency.
        It is only called inside the model training container where sklearn is installed.

        For per_column steps, one transformer tuple is created per column.
        OrdinalEncoder automatically receives unknown_value set to the number of
        unique categories seen in training data (required for handle_unknown).
        """
        import sklearn.preprocessing as sk_pre  # noqa: PLC0415

        encoder_registry = {
            "StandardScaler": sk_pre.StandardScaler,
            "OrdinalEncoder": sk_pre.OrdinalEncoder,
            "OneHotEncoder": sk_pre.OneHotEncoder,
        }

        col_list = X_train.columns.tolist()
        transformers = []
        for step in self.preprocessing:
            EncoderClass = encoder_registry[step.encoder]
            col_indices = [col_list.index(col) for col in step.columns]
            if step.per_column:
                for col, idx in zip(step.columns, col_indices):
                    kwargs = dict(step.kwargs)
                    if step.encoder == "OrdinalEncoder":
                        kwargs["unknown_value"] = X_train[col].nunique()
                    transformers.append(
                        (f"{step.encoder} for {col}", EncoderClass(**kwargs), [idx])
                    )
            else:
                transformers.append(
                    (step.encoder, EncoderClass(**step.kwargs), col_indices)
                )
        return transformers
