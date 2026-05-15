from __future__ import annotations

import importlib

import pandas as pd
import pytest

pytest.importorskip("mudata")

fragments_mod = importlib.import_module("run_dictys_fragments")


def test_p2g_cre_to_bed_rows_accepts_colon_and_hyphen_tail() -> None:
    peak_df = fragments_mod._p2g_cre_to_bed_rows(pd.Series(["chr6-7107480:7107980", "chr1-100-250"]))
    assert peak_df.to_dict(orient="records") == [
        {"chr": "chr6", "start": 7107480, "end": 7107980},
        {"chr": "chr1", "start": 100, "end": 250},
    ]


def test_p2g_cre_to_bed_rows_rejects_bad_cre() -> None:
    with pytest.raises(ValueError, match="Invalid CRE"):
        fragments_mod._p2g_cre_to_bed_rows(pd.Series(["chr6:7107480:7107980"]))


def test_canonical_mdl_peak_loc_normalizes_to_dictys_tssdist_format() -> None:
    assert fragments_mod._canonical_mdl_peak_loc("chr6-7107480:7107980") == "chr6:7107480:7107980"
    assert fragments_mod._canonical_mdl_peak_loc("chr1-100-250") == "chr1:100:250"


def test_build_obs_lookup_and_resolve_obs_name_support_sample_prefix() -> None:
    specs = [fragments_mod.FragmentSpec(path=fragments_mod.pathlib.Path("dummy.tsv.gz"), sample="S1")]
    lookup = fragments_mod._build_obs_lookup(
        obs_names=pd.Index(["S1#AAACCC", "S1#GGGTTT"]),
        fragment_specs=specs,
        transform="auto",
        sample_separator="#",
    )
    assert fragments_mod._resolve_obs_name("AAACCC", specs[0], lookup, {}) == "S1#AAACCC"
    assert fragments_mod._resolve_obs_name("GGGTTT", specs[0], lookup, {}) == "S1#GGGTTT"
    assert fragments_mod._resolve_obs_name("NOTFOUND", specs[0], lookup, {}) is None


def test_barcode_map_has_priority_over_lookup() -> None:
    specs = [fragments_mod.FragmentSpec(path=fragments_mod.pathlib.Path("dummy.tsv.gz"), sample="S1")]
    lookup = {
        ("S1", "AAACCC"): "S1#AAACCC",
    }
    barcode_map = {
        ("S1", "AAACCC"): "manual-cell",
    }
    assert fragments_mod._resolve_obs_name("AAACCC", specs[0], lookup, barcode_map) == "manual-cell"
