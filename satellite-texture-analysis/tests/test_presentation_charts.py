"""The bars must represent the original measured values for every class and patch."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
import numpy as np
import pandas as pd

from modules.pipeline import class_glcm_eigenvalue_vectors
from modules.visualization import (
    EIGEN_LABELS, plot_class_separation, plot_combined_eigenvalue_comparison,
    plot_individual_class_eigenvalues,
)


def heights(ax):
    return [[bar.get_height() for bar in item] for item in ax.containers if isinstance(item, BarContainer)]


def test_signed_class_means_and_all_patch_pages():
    patches = [
        SimpleNamespace(label=label, patch_id=f"{label}_{j}", eigenvalues=np.array([1+j, .2, .1, 0, -.02*j]))
        for label, count in [("one", 5), ("two", 3)] for j in range(count)
    ]
    fig, _ = plot_combined_eigenvalue_comparison(patches)
    ax = fig.axes[0]
    expected = [np.mean([p.eigenvalues for p in patches if p.label == label], axis=0) for label in ["one", "two"]]
    np.testing.assert_allclose(heights(ax), expected)
    assert [t.get_text() for t in ax.get_xticklabels()] == EIGEN_LABELS
    assert ax.get_ylim()[0] < 0
    plt.close(fig)
    for page, members in [(0, patches[:3]), (1, patches[3:5])]:
        fig, _ = plot_individual_class_eigenvalues(patches[:5], page)
        np.testing.assert_allclose(heights(fig.axes[0]), [p.eigenvalues for p in members])
        plt.close(fig)


def test_separation_bars_match_all_six_unique_distances():
    values = np.array([[0,5,3,1],[5,0,6,2],[3,6,0,4],[1,2,4,0]])
    matrix = pd.DataFrame(values, index=list("ABCD"), columns=list("ABCD"))
    fig, _ = plot_class_separation(matrix)
    ax = fig.axes[0]
    assert [p.get_width() for p in ax.patches] == [1,2,3,4,5,6]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A vs D","B vs D","A vs C","C vs D","A vs B","B vs C"]
    assert ax.get_xlim()[0] == 0
    assert ax.get_title() == "Texture distances from combined class GLCMs"
    assert ax.get_xlabel() == "Euclidean distance between class eigenvalue vectors"
    assert fig.texts[0].get_text().splitlines() == [
        "Greatest distance: B vs C (6.0000)", "Smallest distance: A vs D (1.0000)"
    ]
    plt.close(fig)


def test_class_spectrum_comes_from_the_combined_glcm():
    # Each patch has the same spectrum, but the probability mass is in a
    # different cell. Averaging spectra would incorrectly return [1, 0, ...].
    identity = np.eye(5)
    patches = [
        SimpleNamespace(label="A", glcm=np.diag(identity[i]), eigenvalues=identity[0])
        for i in range(3)
    ]
    vectors, labels = class_glcm_eigenvalue_vectors(patches)
    assert labels == ["A"]
    np.testing.assert_allclose(vectors[0], [1 / 3, 1 / 3, 1 / 3, 0, 0])
