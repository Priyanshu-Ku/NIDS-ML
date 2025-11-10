# Models Directory

This directory stores all trained machine learning models.

## Directory Structure:

```
models/
├── random_forest_model.pkl
├── xgboost_model.pkl
├── svm_model.pkl
├── knn_model.pkl
├── logistic_regression_model.pkl
├── naive_bayes_model.pkl
├── lstm_model.h5
├── cnn_model.h5
├── scaler.pkl
├── label_encoder.pkl
└── shap_explainer.pkl
```

## Usage:

### Save a model:

```python
import joblib
joblib.dump(model, 'models/model_name.pkl')
```

### Load a model:

```python
import joblib
model = joblib.load('models/model_name.pkl')
```

Models are automatically saved after training using the `model_training.py` module.
