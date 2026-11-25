"""
Spatially blocked cross-validation for texture feature sets.
Patches that share a geographic block stay in the same fold.
Classifiers: nearest-centroid and RBF SVM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modules.glcm_analysis import build_feature_matrix, zscore_features
from modules.texture_extraction import TexturePatch

CLASS_ORDER = [
    "Water Bodies",
    "Vegetation",
    "Urban / Built-up Area",
    "Bare Soil",
]


def _nearest_centroid_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    labels = []
    for lab in y_train:
        if lab not in labels:
            labels.append(lab)
    centroids = []
    for lab in labels:
        centroids.append(np.mean(X_train[y_train == lab], axis=0))
    C = np.vstack(centroids)
    pred = []
    for row in X_test:
        d = np.linalg.norm(C - row, axis=1)
        pred.append(labels[int(np.argmin(d))])
    return np.asarray(pred)


def _svm_rbf_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> Optional[np.ndarray]:
    try:
        from sklearn.svm import SVC
    except ImportError:
        return None
    # Small-n settings: modest C, scale gamma
    uniq = np.unique(y_train)
    if len(uniq) < 2 or len(y_train) < 4:
        return None
    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    clf.fit(X_train, y_train)
    return np.asarray(clf.predict(X_test))


def _metrics(
    y_true: Sequence[str], y_pred: Sequence[str], class_names: List[str]
) -> Dict[str, Any]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    k = len(class_names)
    idx = {c: i for i, c in enumerate(class_names)}
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    acc = float(np.trace(cm) / max(1, cm.sum()))
    prec = []
    rec = []
    for i, name in enumerate(class_names):
        col = cm[:, i].sum()
        row = cm[i, :].sum()
        tp = cm[i, i]
        prec.append(float(tp / col) if col else 0.0)
        rec.append(float(tp / row) if row else 0.0)
    return {
        "confusion_matrix": cm,
        "overall_accuracy": acc,
        "precision": {class_names[i]: prec[i] for i in range(k)},
        "recall": {class_names[i]: rec[i] for i in range(k)},
        "class_names": class_names,
    }


def blocked_group_folds(
    blocks: List[int], n_splits: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-block-out (or grouped k-fold) using geographic block ids."""
    groups = np.asarray(blocks)
    uniq = []
    for g in groups:
        if g not in uniq:
            uniq.append(g)
    if len(uniq) < 2:
        return []
    k = len(uniq) if n_splits is None else min(int(n_splits), len(uniq))
    # Round-robin if more unique blocks than k
    folds = []
    for i in range(k):
        test_g = set(uniq[i::k] if k < len(uniq) else [uniq[i]])
        if k == len(uniq):
            test_g = {uniq[i]}
        test_idx = np.where(np.isin(groups, list(test_g)))[0]
        train_idx = np.where(~np.isin(groups, list(test_g)))[0]
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def evaluate_feature_set(
    patches: List[TexturePatch],
    feature_set: str,
    include_transition: bool = False,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    X, names, labels, blocks = build_feature_matrix(
        patches, feature_set=feature_set, include_transition=include_transition
    )
    if class_names is None:
        class_names = [c for c in CLASS_ORDER if c in labels]
        for lab in labels:
            if lab not in class_names:
                class_names.append(lab)

    y = np.asarray(labels)
    folds = blocked_group_folds(blocks)
    result: Dict[str, Any] = {
        "feature_set": feature_set,
        "feature_names": names,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_folds": len(folds),
        "nearest_centroid": None,
        "svm_rbf": None,
        "note": "",
    }
    if X.shape[0] < 4 or len(folds) < 2:
        result["note"] = "Not enough spatially distinct blocks/patches for blocked CV."
        return result

    if any(len(np.unique(y[train])) != len(class_names) for train, _ in folds):
        result["note"] = (
            "Blocked CV unavailable: a class occurs in only one independent block group. Supply geographically dispersed patches or change the grid."
        )
        return result
    result["predictions"] = []
    nc_true: List[str] = []
    nc_pred: List[str] = []
    svm_true: List[str] = []
    svm_pred: List[str] = []
    svm_ok = True

    for train_idx, test_idx in folds:
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # Scale using training fold only
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-15, 1.0, sd)
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd
        pred_nc = _nearest_centroid_predict(Ztr, ytr, Zte)
        nc_true.extend(yte.tolist())
        nc_pred.extend(pred_nc.tolist())
        pred_svm = _svm_rbf_predict(Ztr, ytr, Zte)
        for j, index in enumerate(test_idx):
            result["predictions"].append(
                dict(
                    patch_id=patches[index].patch_id,
                    block_id=blocks[index],
                    actual=labels[index],
                    nearest_centroid=str(pred_nc[j]),
                    svm_rbf=str(pred_svm[j]) if pred_svm is not None else None,
                )
            )
        if pred_svm is None:
            svm_ok = False
        else:
            svm_true.extend(yte.tolist())
            svm_pred.extend(pred_svm.tolist())

    result["nearest_centroid"] = _metrics(nc_true, nc_pred, class_names)
    if svm_ok and svm_pred:
        result["svm_rbf"] = _metrics(svm_true, svm_pred, class_names)
    else:
        result["note"] = (
            result["note"]
            + " SVM unavailable (install scikit-learn or too few classes in a fold)."
        ).strip()
    return result


def compare_feature_sets(
    patches: List[TexturePatch],
    include_transition: bool = False,
) -> Dict[str, Any]:
    """Run blocked CV for eigenvalues, classic features, and combined."""
    sets = ["eigenvalues", "classic", "combined"]
    if include_transition:
        sets.append("transition")
    per_set = {}
    best_name = None
    best_acc = -1.0
    best_clf = None
    for name in sets:
        ev = evaluate_feature_set(patches, name, include_transition=include_transition)
        per_set[name] = ev
        for clf_key in ("svm_rbf", "nearest_centroid"):
            block = ev.get(clf_key)
            if not block:
                continue
            acc = float(block["overall_accuracy"])
            if acc > best_acc:
                best_acc = acc
                best_name = f"{name} / {clf_key}"
                best_clf = clf_key
    winners = [
        f"{name} / {clf}"
        for name, ev in per_set.items()
        for clf in ("nearest_centroid", "svm_rbf")
        if ev.get(clf) and ev[clf]["overall_accuracy"] == best_acc
    ]
    return {
        "winners": winners,
        "by_set": per_set,
        "best_feature_set": best_name,
        "best_accuracy": float(best_acc) if best_acc >= 0 else None,
        "best_classifier": best_clf,
        "include_transition": include_transition,
    }


def cv_tables(cv: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Flatten CV results into DataFrames for CSV export."""
    rows = []
    cms = {}
    by_set = cv.get("by_set", {})
    for set_name, ev in by_set.items():
        for clf in ("nearest_centroid", "svm_rbf"):
            block = ev.get(clf)
            if not block:
                continue
            rec = {
                "feature_set": set_name,
                "classifier": clf,
                "overall_accuracy": block["overall_accuracy"],
            }
            for c, p in block["precision"].items():
                rec[f"precision_{c}"] = p
            for c, r in block["recall"].items():
                rec[f"recall_{c}"] = r
            rows.append(rec)
            cms[f"{set_name}_{clf}"] = pd.DataFrame(
                block["confusion_matrix"],
                index=block["class_names"],
                columns=block["class_names"],
            )
    return {"summary": pd.DataFrame(rows), "confusion": cms}
