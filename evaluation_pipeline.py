"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 5 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    try:
        df = pd.read_csv(filepath)
        # Drop customer_id as it is a non-predictive identifier
        if 'customer_id' in df.columns:
            df = df.drop(columns=['customer_id'])
        
        # Separate features (X) and target (y)
        X = df.drop(columns=['churned'])
        y = df['churned']
        return X, y
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def define_models():
    """Define the 5 model configurations to compare.

    Two dummy baselines are included to teach two different lessons:
    most_frequent demonstrates the accuracy inflation problem on imbalanced
    data; stratified shows what random guessing in proportion to class
    frequencies looks like, so F1 carries meaningful signal when comparing.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    preprocessor = build_preprocessor()
    
    configs = {
        "LogReg_default": LogisticRegression(
            C=1.0, random_state=42, max_iter=1000, class_weight="balanced"
        ),
        "LogReg_L1": LogisticRegression(
            C=0.1, penalty="l1", solver="saga", random_state=42, max_iter=1000, class_weight="balanced"
        ),
        "RidgeClassifier": RidgeClassifier(
            alpha=1.0, random_state=42, class_weight="balanced"
        ),
        "Dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "Dummy_stratified": DummyClassifier(strategy="stratified", random_state=42)
    }

    # Wrap each configuration in a pipeline with the preprocessor
    models = {}
    for name, model in configs.items():
        models[name] = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    return models


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    results_list = []
    scoring = ["accuracy", "precision", "recall", "f1"]

    for name, pipeline in models.items():
        # Use cross_validate to get multiple metrics
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        results_list.append({
            "model": name,
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean()
        })

    return pd.DataFrame(results_list)


def final_evaluation(pipeline, X_train, X_test, y_train, y_test):
    """Train a pipeline on full training data and evaluate on the held-out test set."""
    # Fit the pipeline on the full training set
    pipeline.fit(X_train, y_train)
    # Predict on the unseen test set
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    return metrics


def recommend_model(results_df):
    """Print a recommendation based on the results."""
    print("\n=== Model Comparison Table (CV results) ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        # Stratified split to maintain churn ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")
        print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

        # Comparison
        models = define_models()
        if models:
            results = evaluate_models(models, X_train, y_train)
            recommend_model(results)

            # Select best model based on F1-score (excluding dummy baselines)
            real_models_df = results[~results['model'].str.contains('Dummy')]
            best_model_name = real_models_df.loc[real_models_df['f1_mean'].idxmax(), 'model']
            
            print(f"\n---> Selected Best Model: {best_model_name}")
            
            final_metrics = final_evaluation(models[best_model_name], X_train, X_test, y_train, y_test)
            
            print("\n=== Final Test Set Evaluation ===")
            for metric, value in final_metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")

            # Verification: Compare CV Mean F1 with Test F1
            cv_f1 = real_models_df.loc[real_models_df['model'] == best_model_name, 'f1_mean'].values[0]
            print(f"CV Mean F1: {cv_f1:.4f} vs Test F1: {final_metrics['f1']:.4f}")

"""
Task 6: Final Recommendation

I recommend using the RidgeClassifier for predicting customer churn. During our testing, this model was the best at 
finding a balance between overall accuracy and actually catching customers who might leave. While the "Most-frequent 
Dummy" model had a very high accuracy of 83.75%, it is actually useless because it never predicts that a customer will churn.
This proves that for this project, looking at accuracy alone is a mistake.

The RidgeClassifier is much more helpful because it correctly identifies over 65% of churning customers. Even though it
occasionally flags a customer as "churning" when they aren't (low precision), it is still much better than just
guessing randomly. Finally, the model performed even better on our final "unseen" test data (F1: 0.3810) than it did
during training, which gives us confidence that it will work well in the real world.

"""