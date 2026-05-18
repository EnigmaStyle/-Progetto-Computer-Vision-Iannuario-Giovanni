import numpy as np
from src.classical_model import WasteClassifierSVM


def test_fit_predict():
    X = np.random.rand(30, 100).astype(np.float32)
    y = np.array([i % 6 for i in range(30)])
    clf = WasteClassifierSVM()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 30
    assert all(p in range(6) for p in preds)


def test_predict_proba_shape():
    X = np.random.rand(30, 100).astype(np.float32)
    y = np.array([i % 6 for i in range(30)])
    clf = WasteClassifierSVM()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (30, 6)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_save_load(tmp_path):
    X = np.random.rand(30, 100).astype(np.float32)
    y = np.array([i % 6 for i in range(30)])
    clf = WasteClassifierSVM()
    clf.fit(X, y)
    path = str(tmp_path / "svm_test.joblib")
    clf.save(path)
    clf2 = WasteClassifierSVM.load(path)
    preds1 = clf.predict(X)
    preds2 = clf2.predict(X)
    assert np.array_equal(preds1, preds2)
