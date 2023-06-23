# Pipeline

The pipelines can be found in the [`pipelines/src/pipelines/`](../pipelines/src/pipelines) folder.
Find below an in-depth explanation of the training and prediction pipeline.

## Training pipeline 

The training pipeline automates common tasks in a Data Science lifecycle including:
- gathering & preprocessing data
- training & evaluating the model
- selecting the best model for deployment

As part of this pipeline, the model is trained by creating a [custom training job](https://cloud.google.com/vertex-ai/docs/training/create-custom-job) with added flexibility for `machine_type`, `replica_count`, `accelerator_type` among other machine configurations.
The training logic is contained in [`pipelines/assets/train_model.py`](../pipelines/assets/train_model.py) which trains an [XGBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor) using a [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

Within the scikit-learn pipeline, the model training step is preceded by a preprocessing phase where different transformations are applied to the training and evaluation data using scikit-learn preprocessing functions:

![Training process](./images/xgboost_architecture.png)

### Preprocessing

The 3 data transformation steps considered in the training script are:

|Encoder|Description|Features|
|:----|:----|:----|
|[StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)|Centering and scaling numerical values|   `dayofweek`, `hourofday`, `trip_distance`, `trip_miles`, `trip_seconds`|
|[OneHotEncoder()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)|Encoding a chosen subset of categorical features as a one-hot numeric array|`payment_type`, new/unknown values in categorial features are represented as zeroes everywhere in the one-hot numeric array|
|[OrdinalEncoder()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)|Encoding a chosen subset of categorical features as an integer array|`company`, new/unknown values in categorical features are assigned to an integer equal to the number of categories for that feature in the training set|

More processing steps can be included to the pipeline. 
For more details, see the [official documentation](https://scikit-learn.org/stable/modules/preprocessing.html). 
Ensure that these additional pre-processing steps can handle new/unknown values in test data.

### Model training

In our example implementation, we have a regression problem of predicting the total fare of a taxi trip in Chicago.

You can specify different hyperparameters through the `model_params` argument of `train_xgboost_model`, including:
  - `Booster`: the type of booster (`gbtree` is a tree based booster used by default).
  - `max_depth`: the depth of each tree.
  - `Objective`: equivalent to the loss function (squared loss, `reg:squarederror`, is the default).
  - `min_split_loss`: the minimum loss reduction required to make a further partition on a leaf node of the tree.

More hyperparameters can be used to customize your training. 
For more details consult the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)


### Model evaluation

Once the model is trained, the training script evaluates the model on a held-out test dataset. 
The resulting metrics are used for comparing the newly trained model to any existing models.
If you are working with larger test data, it is more efficient to use [`ModelBatchPredictOp`](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-0.2.1/google_cloud_pipeline_components.aiplatform.html).

### Results

Two model artifacts are generated when we run the training job: 
  - `model`: The model is exported to GCS file as a [joblib](https://joblib.readthedocs.io/en/latest/why.html#benefits-of-pipelines) object.
  - `metrics`: The evaluation metrics are exported to GCS as JSON file.
