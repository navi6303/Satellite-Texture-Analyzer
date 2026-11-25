import numpy as np
import pytest

from modules.glcm_analysis import (
    classic_glcm_features,
    compute_eigenvalues,
    fisher_ratios,
)
from modules.patch_sampling import sample_layout
from modules.snip_tool import decode_selection
from modules.visualization import _annotate_bars
from modules.workflow import Settings


def test_exact_is_default_and_keeps_each_rectangle_without_rng():
    assert Settings().mode == "global_equalized_fixed5"
    assert Settings().exact_boxes
    manual = {"a": [dict(x=3, y=4, w=22, h=11)], "b": [dict(x=31, y=15, w=12, h=18)]}
    for seed in (0, 555):
        result = sample_layout(
            ["a", "b"], manual, {}, 5, 64, 256, seed, 60, 60, exact=True
        )
        assert result == manual
    with pytest.raises(ValueError, match="bounding box"):
        sample_layout(
            ["a", "b"], {"a": manual["a"]}, {}, 5, 64, 256, 0, 60, 60, exact=True
        )
    box, ok, _ = decode_selection(dict(x=3, y=4, w=22, h=11), 60, 60)
    assert ok and box == manual["a"][0]


def test_handbook_features_analytic_two_levels():
    P = np.zeros((5, 5))
    P[0, 1] = P[1, 0] = 0.25
    P[1, 1] = 0.5
    f = classic_glcm_features(P)
    assert f == pytest.approx(
        dict(
            asm=0.375,
            energy=np.sqrt(0.375),
            entropy=1.5,
            contrast=0.5,
            dissimilarity=0.5,
            homogeneity=0.75,
            correlation=-1 / 3,
            glcm_mean=0.75,
            glcm_variance=0.1875,
            max_probability=0.5,
            glcm_trace=0.5,
            diagonal_share=0.5,
        )
    )
    assert sum(compute_eigenvalues(P)) == pytest.approx(0.5)


def test_same_trace_different_lambda_and_no_singleton_fisher():
    assert np.trace(np.eye(5) / 5) == pytest.approx(1)
    assert compute_eigenvalues(np.eye(5) / 5)[0] == pytest.approx(0.2)
    concentrated = np.diag([1.0, 0, 0, 0, 0])
    assert compute_eigenvalues(concentrated)[0] == pytest.approx(1)
    assert fisher_ratios(np.array([[1], [3]]), ["a", "b"], ["x"]).empty


def test_chart_text_attached_to_upper_cap():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    bars = ax.bar([0, 1], [0.2, -0.1])
    _annotate_bars(ax, bars, [0.2, -0.1], errors=[0.05, 0.2])
    assert [t.xy[1] for t in ax.texts] == pytest.approx([0.25, 0.1])
    assert all(t.get_position()[1] > 0 and t.get_rotation() == 0 for t in ax.texts)
    plt.close(fig)
