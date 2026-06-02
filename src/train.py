import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def generate_mock_data(samples=1000):
    X = []
    y = []

    for _ in range(samples):
        bg_noise = np.random.uniform(100, 5000, 100)
        X.append(bg_noise)
        y.append("BACKGROUND")

        uav_noise = np.random.uniform(100, 5000, 100)
        uav_noise[:15] += np.random.uniform(100000, 800000, 15)
        uav_noise[40:50] += np.random.uniform(50000, 200000, 10)
        X.append(uav_noise)
        y.append("UAV")

    return np.array(X), np.array(y)


def train_model():
    print("Generating synthetic data...")
    X, y = generate_mock_data(2000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_v1.joblib"

    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    train_model()
