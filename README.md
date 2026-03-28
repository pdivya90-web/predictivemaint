
---
tags:
- tabular-classification
- scikit-learn
- xgboost
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: XGBoost Predictive Maintenance Model
  results:
    - task:
        type: tabular-classification
        name: Tabular Classification
      dataset:
        name: Engine Sensor Data
        type: custom
      metrics:
        - type: accuracy
          value: 0.6647
        - type: precision
          value: 0.6479
        - type: recall
          value: 0.6647
        - type: f1
          value: 0.6358
---
# XGBoost Predictive Maintenance Model

This model is an XGBoost Classifier trained to predict engine condition (healthy or failing) based on sensor data. It's part of a predictive maintenance system.

## Model Details

-   **Model Type:** XGBoost Classifier
-   **Task:** Binary Classification (Engine Condition: 0 = Healthy, 1 = Failing)
-   **Best Parameters (from GridSearchCV):**
    ```json
    {'objective': 'binary:logistic', 'base_score': None, 'booster': None, 'callbacks': None, 'colsample_bylevel': None, 'colsample_bynode': None, 'colsample_bytree': None, 'device': None, 'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': 'logloss', 'feature_types': None, 'feature_weights': None, 'gamma': None, 'grow_policy': None, 'importance_type': None, 'interaction_constraints': None, 'learning_rate': 0.01, 'max_bin': None, 'max_cat_threshold': None, 'max_cat_to_onehot': None, 'max_delta_step': None, 'max_depth': 6, 'max_leaves': None, 'min_child_weight': None, 'missing': nan, 'monotone_constraints': None, 'multi_strategy': None, 'n_estimators': 200, 'n_jobs': None, 'num_parallel_tree': None, 'random_state': 42, 'reg_alpha': None, 'reg_lambda': None, 'sampling_method': None, 'scale_pos_weight': None, 'subsample': None, 'tree_method': None, 'validate_parameters': None, 'verbosity': None, 'use_label_encoder': False}
    ```

## Training Data

The model was trained on the `Engine Sensor Data` dataset, which includes various engine sensor readings such as RPM, oil pressure, fuel pressure, coolant pressure, and temperatures.

-   **Features (X_train):** `X_train.csv` (scaled numerical features)
-   **Target (y_train):** `y_train.csv` (Engine Condition)

## Evaluation Results

The model was evaluated on a held-out test set (`X_test.csv`, `y_test.csv`).

-   **Accuracy:** 0.6647
-   **Precision (weighted):** 0.6479
-   **Recall (weighted):** 0.6647
-   **F1-score (weighted):** 0.6358

## Usage

This model can be used to predict the `Engine Condition` for new engine sensor data. It was trained using `scikit-learn` and `xgboost` libraries.

To load and use the model:

```python
import joblib

# Load the model
model = joblib.load('best_xgboost_model.joblib')

# Make predictions (example with dummy data)
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# new_data = scaler.fit_transform([[750, 3.0, 7.0, 2.5, 78.0, 80.0]]) # Example scaled data matching training features
# prediction = model.predict(new_data)
# print("Predicted Engine Condition: [predicted value]") # Modified to avoid NameError
```
