"""Autograder tests for Integration 5A — ML Evaluation Pipeline."""

import pytest
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation_pipeline import (load_and_prepare, build_preprocessor,
                                  define_models, evaluate_models,
                                  final_evaluation)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "telecom_churn.csv"
)


# ── Data Loading ──────────────────────────────────────────────────────────

def test_data_loaded():
    """load_and_prepare returns (X, y) with correct shape and no target leak."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None"
    X, y = result
    assert X.shape[0] > 1000, f"Expected >1000 rows, got {X.shape[0]}"
    assert "churned" not in X.columns, "Target 'churned' should not be in features"
    assert len(y) == len(X), "X and y must have the same number of rows"
    assert set(y.unique()).issubset({0, 1}), "Target should be binary (0/1)"


def test_features_include_numeric_and_categorical():
    """Features should include both numeric and categorical columns."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, _ = result
    numeric_expected = {"tenure", "monthly_charges", "total_charges", "num_support_calls"}
    categorical_expected = {"contract_type", "internet_service", "payment_method"}
    assert numeric_expected.issubset(set(X.columns)), (
        f"Missing numeric features: {numeric_expected - set(X.columns)}"
    )
    assert categorical_expected.issubset(set(X.columns)), (
        f"Missing categorical features: {categorical_expected - set(X.columns)}"
    )


# ── Preprocessor ──────────────────────────────────────────────────────────

def test_preprocessor_built():
    """build_preprocessor returns a ColumnTransformer with fit_transform."""
    prep = build_preprocessor()
    assert prep is not None, "build_preprocessor returned None"
    assert hasattr(prep, "fit_transform"), "Preprocessor must have fit_transform"


def test_preprocessor_transforms_data():
    """Preprocessor should transform the data without error."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, _ = result
    prep = build_preprocessor()
    transformed = prep.fit_transform(X)
    assert transformed is not None, "fit_transform returned None"
    assert transformed.shape[0] == X.shape[0], "Row count must be preserved"
    # OneHotEncoder expands categoricals — output should have more columns
    assert transformed.shape[1] > X.shape[1], (
        "Transformed data should have more columns than input (OneHotEncoder expansion)"
    )


# ── Model Definitions ────────────────────────────────────────────────────

def test_models_defined():
    """define_models returns at least 5 named Pipelines with fit methods."""
    models = define_models()
    assert models is not None, "define_models returned None"
    assert len(models) >= 5, (
        f"Expected >= 5 models (3 real + 2 dummy baselines: most_frequent "
        f"and stratified), got {len(models)}. See Task 3 in the integration "
        f"guide for the required model configurations."
    )
    for name, pipe in models.items():
        assert hasattr(pipe, "fit"), f"Model '{name}' must have fit method"
        assert hasattr(pipe, "predict"), f"Model '{name}' must have predict method"


def test_models_include_dummy_baseline():
    """A DummyClassifier baseline must be among the defined models."""
    models = define_models()
    assert models is not None
    model_names_lower = [n.lower() for n in models.keys()]
    assert any("dummy" in n or "baseline" in n for n in model_names_lower), (
        "Models must include a DummyClassifier baseline"
    )


def test_models_include_stratified_dummy():
    """Models must include a DummyClassifier with strategy='stratified'.

    The most_frequent dummy is structurally degenerate on imbalanced data
    (it never predicts the positive class, so F1 = 0 trivially). The
    stratified dummy provides a non-trivial F1 baseline (~positive class
    rate) against which real models must demonstrate meaningful signal.
    Both dummies are required — see Task 3 in the integration guide.
    """
    from sklearn.dummy import DummyClassifier
    models = define_models()
    assert models is not None

    has_stratified = False
    for pipe in models.values():
        for _, step in pipe.steps:
            if isinstance(step, DummyClassifier) and step.strategy == "stratified":
                has_stratified = True
                break
        if has_stratified:
            break

    assert has_stratified, (
        "Models must include a DummyClassifier(strategy='stratified') in "
        "addition to the most_frequent dummy. The stratified dummy provides "
        "the non-trivial F1 baseline — see Task 3 in the integration guide "
        "for why both dummies are required."
    )


def test_models_are_pipelines():
    """Each model should be a Pipeline (preprocessor + estimator)."""
    from sklearn.pipeline import Pipeline
    models = define_models()
    assert models is not None
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), (
            f"Model '{name}' should be a Pipeline, got {type(pipe).__name__}"
        )


# ── Evaluation ────────────────────────────────────────────────────────────

def test_evaluation_runs():
    """evaluate_models returns a DataFrame with expected columns and rows."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, y = result
    models = define_models()
    assert models is not None

    results_df = evaluate_models(models, X, y)
    assert results_df is not None, "evaluate_models returned None"
    assert isinstance(results_df, pd.DataFrame), "Results must be a DataFrame"
    assert len(results_df) >= 5, (
        f"Expected >= 5 rows (3 real models + 2 dummy baselines), "
        f"got {len(results_df)}"
    )


def test_evaluation_has_required_columns():
    """Results DataFrame must contain mean columns for key metrics."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, y = result
    models = define_models()
    results_df = evaluate_models(models, X, y)
    assert results_df is not None

    required_cols = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    for col in required_cols:
        assert col in results_df.columns, f"Missing column: {col}"


def test_evaluation_metrics_are_reasonable():
    """Metric values should be between 0 and 1, and real models beat baseline."""
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, y = result
    models = define_models()
    results_df = evaluate_models(models, X, y)
    assert results_df is not None

    for col in ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]:
        if col in results_df.columns:
            values = results_df[col].values
            assert all(0 <= v <= 1 for v in values), (
                f"All {col} values should be in [0, 1]"
            )

    # Both dummies must be present and the real models must beat the best
    # dummy F1 by a non-trivial margin. The most_frequent dummy gives F1=0
    # structurally (never predicts the positive class), so the F1 comparison
    # is only meaningful when the stratified dummy is also evaluated —
    # stratified dummy F1 ≈ positive class rate, giving real models a
    # non-trivial baseline to beat.
    if "f1_mean" in results_df.columns and "model" in results_df.columns:
        dummy_rows = results_df[
            results_df["model"].str.lower().str.contains("dummy|baseline")
        ]
        real_rows = results_df[
            ~results_df["model"].str.lower().str.contains("dummy|baseline")
        ]
        assert len(dummy_rows) >= 2, (
            f"Results should include both dummy baselines (most_frequent and "
            f"stratified), got {len(dummy_rows)} dummy row(s). See Task 3 "
            f"in the integration guide."
        )
        assert len(real_rows) > 0, "Results should include real (non-dummy) models"
        best_real = real_rows["f1_mean"].max()
        best_dummy = dummy_rows["f1_mean"].max()
        assert best_dummy > 0, (
            "Best dummy F1 should be > 0. The most_frequent dummy gives F1=0 "
            "trivially; a positive best_dummy means the stratified dummy was "
            "included and evaluated correctly. If this fails, you are missing "
            "the stratified dummy — see Task 3 in the integration guide."
        )
        assert best_real > best_dummy, (
            "At least one real model should beat the best dummy baseline on F1. "
            "With the stratified dummy as a non-trivial floor (F1 ≈ positive "
            "class rate), this is a genuine test of whether your real models "
            "have learned meaningful signal from the features."
        )


# ── Final Evaluation on Held-Out Test Set ────────────────────────────────

def test_final_evaluation_returns_metrics():
    """final_evaluation trains on X_train and evaluates on X_test, returning metrics.

    Task 5 requires the learner to take the best model from CV and evaluate
    it on the held-out test set — this test verifies the final_evaluation
    function is implemented and returns the correct dict structure.
    """
    from sklearn.model_selection import train_test_split
    result = load_and_prepare(DATA_PATH)
    assert result is not None, "load_and_prepare returned None — implement it first"
    X, y = result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = define_models()
    assert models is not None

    # Pick a real (non-dummy) model to exercise final_evaluation
    real_models = {
        n: p for n, p in models.items()
        if "dummy" not in n.lower() and "baseline" not in n.lower()
    }
    assert len(real_models) > 0, (
        "Models dict must include at least one real (non-dummy) model"
    )
    test_pipe = list(real_models.values())[0]

    metrics = final_evaluation(test_pipe, X_train, X_test, y_train, y_test)
    assert metrics is not None, "final_evaluation returned None"
    assert isinstance(metrics, dict), (
        f"final_evaluation should return a dict, got {type(metrics).__name__}"
    )
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in metrics, (
            f"final_evaluation result missing key '{key}'. "
            f"See Task 5 in the integration guide."
        )
        assert 0 <= metrics[key] <= 1, (
            f"final_evaluation metric {key}={metrics[key]} should be in [0, 1]"
        )
