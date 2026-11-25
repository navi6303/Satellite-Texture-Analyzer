"""Verify that quantized histogram bars use the selected patch's valid pixels."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from modules.visualization import plot_quantized_patch_histogram


def test_patch_histogram_counts_and_labels_all_five_levels():
    patch = SimpleNamespace(
        label="Water",
        patch_id="patch_07",
        quantized=np.array([[0, 1, 1], [3, 4, 4]], dtype=np.uint8),
        valid_mask=np.array([[False, True, True], [True, False, True]]),
    )

    fig, png = plot_quantized_patch_histogram(patch)
    try:
        ax = fig.axes[0]
        assert [bar.get_height() for bar in ax.patches] == [0, 2, 0, 1, 1]
        assert [label.get_text() for label in ax.texts] == ["0", "2", "0", "1", "1"]
        assert [label.get_text() for label in ax.get_xticklabels()] == ["0", "1", "2", "3", "4"]
        assert ax.get_title() == "Water · patch_07"
        assert ax.get_xlabel() == "Gray level"
        assert ax.get_ylabel() == "Number of pixels"
        assert png.startswith(b"\x89PNG")
    finally:
        plt.close(fig)
