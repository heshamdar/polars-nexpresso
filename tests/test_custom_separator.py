"""
The path separator is configurable, and nothing may assume it is ``"."``.

``HierarchicalPacker(granularity_separator=...)`` accepts any string, so a
project whose field names contain dots can use ``"__"`` or ``"::"`` instead.
Every path is then built and parsed with that separator, and the escape
convention (``escape_char`` before a literal separator) has to hold for it too.

This was not true before 0.10.0: both path splitters compared **one character**
at a time (``path[i] == separator``), so any separator longer than one character
never matched and a path came back as a single unsplit component. Column
ownership is resolved by splitting, so `HierarchyView` could not route anything
at all under a multi-character separator. These tests run the same checks across
several separators, including one-character, multi-character, and one containing
a regex metacharacter and spaces.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

SPEC = HierarchySpec.from_levels(
    LevelSpec(name="region", id_fields=["id"]),
    LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
    LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
)

# One character (the default), several characters, a regex metacharacter, and
# one with spaces -- each of which breaks a different naive implementation.
SEPARATORS = [".", "__", "::", "/", "|", " -> "]


@pytest.fixture(params=SEPARATORS, ids=lambda s: repr(s))
def packer(request) -> HierarchicalPacker:
    return HierarchicalPacker(SPEC, granularity_separator=request.param)


@pytest.fixture
def flat(packer: HierarchicalPacker) -> pl.DataFrame:
    j = packer.join_path
    return pl.DataFrame(
        {
            j(["region", "id"]): [1, 1, 1, 2],
            j(["region", "name"]): ["north", "north", "north", "south"],
            j(["region", "store", "region_id"]): [1, 1, 1, 2],
            j(["region", "store", "id"]): [10, 10, 11, 20],
            j(["region", "store", "discount"]): [0.0, 0.0, 0.5, 0.1],
            j(["region", "store", "sale", "store_id"]): [10, 10, 11, 20],
            j(["region", "store", "sale", "id"]): [100, 101, 110, 200],
            j(["region", "store", "sale", "amount"]): [5.0, 6.0, 7.0, 8.0],
        }
    )


@pytest.fixture
def view(flat: pl.DataFrame, packer: HierarchicalPacker) -> HierarchyView:
    return HierarchyView.from_frame(flat, packer)


class TestPathRoundTrip:
    """join_path and split_path must be inverses for any separator."""

    def test_plain_components(self, packer: HierarchicalPacker):
        parts = ["region", "store", "sale", "amount"]
        assert packer.split_path(packer.join_path(parts)) == parts

    def test_field_containing_the_separator(self, packer: HierarchicalPacker):
        """The case the escape convention exists for."""
        sep = packer.separator
        parts = ["region", "store", f"net{sep}sales"]
        joined = packer.join_path(parts)
        assert packer.split_path(joined) == parts
        # It must not split at the escaped occurrence.
        assert len(packer.split_path(joined)) == 3

    def test_field_containing_the_escape_character(self, packer: HierarchicalPacker):
        parts = ["region", "store", f"back{packer.escape_char}slash"]
        assert packer.split_path(packer.join_path(parts)) == parts

    def test_separator_appears_in_every_level_path(self, packer: HierarchicalPacker):
        sep = packer.separator
        paths = [meta.path for meta in packer._levels_meta]  # noqa: SLF001
        assert paths == ["region", f"region{sep}store", f"region{sep}store{sep}sale"]


class TestPackerOperations:
    """The packing API is separator-agnostic end to end."""

    def test_pack_unpack_round_trip(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        packed = packer.pack(flat, "region")
        assert_frame_equal(
            packer.unpack(packed, "sale").sort(packer.join_path(["region", "store", "sale", "id"])),
            flat.sort(packer.join_path(["region", "store", "sale", "id"])),
            check_column_order=False,
        )

    def test_infer_current_level(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        for level in ("region", "store", "sale"):
            assert packer.infer_current_level(packer.pack(flat, level)) == level

    def test_normalize_denormalize(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        tables = packer.normalize(flat.lazy())
        assert set(tables) == {"region", "store", "sale"}
        rebuilt = packer.denormalize(tables).collect()  # type: ignore[union-attr]
        assert_frame_equal(
            rebuilt.sort(packer.join_path(["region", "id"])),
            packer.pack(flat, "region").sort(packer.join_path(["region", "id"])),
        )

    def test_promote_attribute(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        promoted = packer.promote_attribute(
            packer.pack(flat, "store"),
            "amount",
            from_level="sale",
            to_level="store",
            agg="sum",
            alias="revenue",
        )
        assert packer.join_path(["region", "store", "revenue"]) in promoted.columns


class TestViewOperations:
    """HierarchyView resolves ownership by splitting, so it is the strictest test."""

    def test_level_of_resolves(self, view: HierarchyView, packer: HierarchicalPacker):
        j = packer.join_path
        assert view.level_of(j(["region", "name"])) == "region"
        assert view.level_of(j(["region", "store", "discount"])) == "store"
        assert view.level_of(j(["region", "store", "sale", "amount"])) == "sale"

    def test_key_columns(self, view: HierarchyView, packer: HierarchicalPacker):
        j = packer.join_path
        assert view.key_columns("sale") == [
            j(["region", "id"]),
            j(["region", "store", "id"]),
            j(["region", "store", "sale", "id"]),
        ]

    def test_level_joins_the_axis(self, view: HierarchyView, packer: HierarchicalPacker):
        columns = view.level("sale").collect_schema().names()
        assert packer.join_path(["region", "name"]) in columns
        assert view.level("sale").collect().height == 4
        assert view.level("region").collect().height == 2

    def test_filter_routes_and_cascades(self, view: HierarchyView, packer: HierarchicalPacker):
        amount = packer.join_path(["region", "store", "sale", "amount"])
        heights = {
            name: lf.collect().height
            for name, lf in view.filter(pl.col(amount) > 5).tables().items()
        }
        assert heights == {"region": 2, "store": 3, "sale": 3}

    def test_rollup_on_key_columns(self, view: HierarchyView, packer: HierarchicalPacker):
        amount = packer.join_path(["region", "store", "sale", "amount"])
        rolled = (
            view.tables()["sale"]
            .group_by(view.key_columns("region"))
            .agg(pl.col(amount).sum().alias("total"))
            .collect()
        )
        assert sorted(rolled["total"].to_list()) == [8.0, 18.0]

    def test_with_level_accepts_a_qualified_name(
        self, view: HierarchyView, packer: HierarchicalPacker
    ):
        amount = packer.join_path(["region", "store", "sale", "amount"])
        doubled = packer.join_path(["region", "store", "sale", "doubled"])
        widened = view.with_level(
            "sale", lambda lf: lf.with_columns((pl.col(amount) * 2).alias(doubled))
        )
        assert doubled in widened.level("sale").collect_schema().names()

    def test_with_level_still_rejects_an_unqualified_name(
        self, view: HierarchyView, packer: HierarchicalPacker
    ):
        """The naming guard must key off the configured separator, not a dot."""
        amount = packer.join_path(["region", "store", "sale", "amount"])
        with pytest.raises(ValueError, match="name no level in this view"):
            view.with_level("sale", lambda lf: lf.with_columns(pl.col(amount).alias("doubled")))

    def test_nested_round_trip(self, view: HierarchyView, flat: pl.DataFrame, packer):
        assert_frame_equal(
            view.nested().collect().sort(packer.join_path(["region", "id"])),
            packer.pack(flat, "region").sort(packer.join_path(["region", "id"])),
        )
