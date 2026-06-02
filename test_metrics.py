"""Unit tests for GovTech-Bench metric implementations."""

import pytest
from govtechbench.metrics import (
    classification_metrics,
    fraud_detection_metrics,
    ner_metrics,
    LatencyTracker,
)


def test_classification_perfect():
    y = ["valid", "fraudulent", "duplicate"]
    result = classification_metrics(y, y)
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0


def test_classification_all_wrong():
    y_true = ["valid", "valid", "valid"]
    y_pred = ["fraudulent", "fraudulent", "fraudulent"]
    result = classification_metrics(y_true, y_pred)
    assert result["accuracy"] == 0.0


def test_fraud_fpr_zero_when_perfect():
    y_true = ["valid", "fraudulent", "valid"]
    y_pred = ["valid", "fraudulent", "valid"]
    result = fraud_detection_metrics(y_true, y_pred)
    assert result["false_positive_rate"] == 0.0


def test_fraud_fpr_nonzero():
    y_true = ["valid", "valid"]
    y_pred = ["fraudulent", "valid"]
    result = fraud_detection_metrics(y_true, y_pred)
    assert result["false_positive_rate"] == 0.5


def test_ner_perfect():
    fields = ["employee_name", "doc_type"]
    y_true = [{"employee_name": "Jane Doe", "doc_type": "US_Passport"}]
    y_pred = [{"employee_name": "Jane Doe", "doc_type": "US_Passport"}]
    result = ner_metrics(y_true, y_pred, fields=fields)
    assert result["employee_name"]["f1"] == 1.0
    assert result["__macro_f1__"] == 1.0


def test_ner_partial():
    fields = ["employee_name", "doc_type"]
    y_true = [{"employee_name": "Jane Doe", "doc_type": "US_Passport"}]
    y_pred = [{"employee_name": "Jane Doe", "doc_type": "Drivers_License"}]
    result = ner_metrics(y_true, y_pred, fields=fields)
    assert result["employee_name"]["f1"] == 1.0
    assert result["doc_type"]["f1"] == 0.0


def test_latency_tracker():
    import time
    tracker = LatencyTracker()
    for _ in range(3):
        with tracker:
            time.sleep(0.005)
    assert tracker.mean_ms > 0
    assert tracker.p95_ms >= tracker.mean_ms
    assert tracker.summary["n_calls"] == 3
