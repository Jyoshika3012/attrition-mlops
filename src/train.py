import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from preprocess import load_and_preprocess

def train(n_estimators=100, max_depth=6):
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    mlflow.set_experiment("attrition-experiment")

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)

        # Log everything to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Save model locally
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "models/model.pkl")

        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")

if __name__ == "__main__":
    train(n_estimators=100, max_depth=6)
    train(n_estimators=200, max_depth=8)
    train(n_estimators=300, max_depth=10)