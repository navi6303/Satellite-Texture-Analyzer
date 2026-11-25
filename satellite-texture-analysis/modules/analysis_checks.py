"""Run pytest against this analysis's actual masked patches, spectra and Fisher table."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def run_analysis_checks(patches, df_fisher, distance=1):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for i, patch in enumerate(patches):
            np.savez(
                folder / f"patch_{i:04d}.npz",
                patch_id=patch.patch_id,
                matrix=patch.glcm,
                eig=patch.eigenvalues,
                q=patch.quantized,
                native_shape=[patch.height, patch.width],
                raw_counts=[d["raw_counts"] for d in patch.extra["directional"]],
                mode=patch.extra.get("quantization_mode", ""),
                equalized=patch.image_8bit,
                mask=patch.valid_mask,
                pairs=patch.horizontal_valid_pairs,
                label=patch.label,
                distances=patch.extra["distances"],
                angles=patch.extra["angles_deg"],
            )
        np.save(folder / "fisher.npy", df_fisher["fisher_ratio"].to_numpy(dtype=float))
        env = dict(
            os.environ, TEXTURE_CHECK_DIR=tmp, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--color=no",
                "-p",
                "no:cacheprovider",
                "tests/test_run_invariants.py",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
    return dict(
        passed=proc.returncode == 0,
        exit_code=proc.returncode,
        patches=len(patches),
        output=(proc.stdout + proc.stderr).strip(),
    )


def checks_markdown(bundle):
    outcome = "PASS" if bundle["passed"] else "FAIL"
    return f"**{outcome}** · {bundle['patches']} patches\n\n```text\n{bundle['output']}\n```"
