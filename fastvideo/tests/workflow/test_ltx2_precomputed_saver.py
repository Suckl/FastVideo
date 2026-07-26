# SPDX-License-Identifier: Apache-2.0
"""Path-safety tests for LTX-2 precomputed artifact output."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastvideo.workflow.preprocess.preprocess_workflow_ltx2_t2v import (
    LTX2PrecomputedSaver, )


@pytest.mark.parametrize(
    "sample_name",
    [
        "../../target",
        "../target",
        "/absolute/target",
        r"C:\absolute\target",
        r"C:drive-relative",
        r"nested\..\target",
        "",
    ],
)
def test_ltx2_saver_rejects_unsafe_sample_name(
    tmp_path: Path,
    sample_name: str,
) -> None:
    saver = LTX2PrecomputedSaver(tmp_path / ".precomputed")

    with pytest.raises(ValueError):
        saver._to_rel_pt_path(sample_name)


def test_ltx2_saver_rejects_output_path_outside_root(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "latents").resolve()
    root.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        LTX2PrecomputedSaver._contained_output_path(
            root,
            Path("../outside.pt"),
        )


def test_ltx2_saver_accepts_nested_relative_path(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "latents").resolve()
    root.mkdir()

    output = LTX2PrecomputedSaver._contained_output_path(
        root,
        Path("nested/sample.pt"),
    )

    assert output == root / "nested" / "sample.pt"
