#standard library
import json
from datetime import datetime
from pathlib import Path

#external library
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def main():
    #load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    #create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )
    '''
    Check shape
    print(X_train.shape)
    print(y_test.shape)
    '''

    #Create Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ])

    #fit data and test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)

    print(acc_score)
    print(classification_report(y_test, y_pred))

    #save model
    ROOT  = Path(__file__).resolve().parents[1]
    MODELS_DIR = ROOT / 'models'
    MODELS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    timestamp_model_path = MODELS_DIR / f'model_{timestamp}.joblib'
    stable_model_path = MODELS_DIR / f'model.joblib'
    metadata_path = MODELS_DIR / (f'metadata_{timestamp}.json')

    joblib.dump(pipeline, timestamp_model_path)
    joblib.dump(pipeline, stable_model_path)

    #save metadata
    metadata = {
        "accuracy": acc_score,
        "timestamp": datetime.utcnow().isoformat(),
        "testsize": 0.2,
        "random_state": 77,
        "model": "LogisticRegression",
        "scaler": "StandardScaler",
        "max_iter": 1000,
        "n_features": len(data.feature_names),
        "n_samples": len(data.data)
    }

    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile, indent=2)

if __name__ == "__main__":
    main()