"""
``HierarchyView.level`` — the view's main entry point.

``level(g)`` joins the root → ``g`` axis and hands back an ordinary
``LazyFrame``. Two properties make that worth doing instead of asking the caller
to write the join:

* it agrees with ``unpack(packed, g)`` column for column, so the frame is the
  one the packed path would have produced; and
* asking for the whole axis is close to free when you do not read it, because
  projection and predicate pushdown prune the levels you ignore.

The second is the load-bearing claim of the design — it is why ``level`` does
not need to be told which columns the caller intends to use — so it is asserted
here against real Parquet scans rather than assumed.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nexpresso import HierarchicalPacker, HierarchySpec, HierarchyView, LevelSpec

ROOT = "region"
REGION_ID = "region.id"
REGION_NAME = "region.name"
STORE_ID = "region.store.id"
DISCOUNT = "region.store.discount"
SALE_ID = "region.store.sale.id"
AMOUNT = "region.store.sale.amount"

LEVELS = ["region", "store", "sale"]


@pytest.fixture
def packer() -> HierarchicalPacker:
    return HierarchicalPacker(
        HierarchySpec.from_levels(
            LevelSpec(name="region", id_fields=["id"]),
            LevelSpec(name="store", id_fields=["id"], parent_keys=["region_id"]),
            LevelSpec(name="sale", id_fields=["id"], parent_keys=["store_id"]),
        )
    )


@pytest.fixture
def flat() -> pl.DataFrame:
    """Uneven fan-out, so a broken join cannot produce a plausible row count."""
    rows = [
        (0, "north", 0, 0.00, 0, 10.0),
        (0, "north", 0, 0.00, 1, 25.0),
        (0, "north", 1, 0.10, 2, 55.0),
        (1, "south", 2, 0.25, 3, 70.0),
        (1, "south", 2, 0.25, 4, 15.0),
        (2, "east", 3, 0.05, 5, 30.0),
        (2, "east", 4, 0.20, 6, 45.0),
    ]
    return pl.DataFrame(
        {
            REGION_ID: [r[0] for r in rows],
            REGION_NAME: [r[1] for r in rows],
            STORE_ID: [r[2] for r in rows],
            DISCOUNT: [r[3] for r in rows],
            SALE_ID: [r[4] for r in rows],
            AMOUNT: [r[5] for r in rows],
        }
    )


@pytest.fixture
def view(flat: pl.DataFrame, packer: HierarchicalPacker) -> HierarchyView:
    return HierarchyView.from_frame(flat, packer)


class TestAgreesWithTheHandWrittenJoin:
    """level(g) is the join the caller would otherwise write."""

    @pytest.mark.parametrize("target", LEVELS)
    def test_matches_the_explicit_axis_join(
        self, target: str, view: HierarchyView, packer: HierarchicalPacker
    ):
        tables = view.tables()
        axis = packer.get_axis(target)
        want = tables[axis[0]]
        for parent, child in zip(axis[:-1], axis[1:]):
            keys = [
                c for c in view.key_columns(parent) if c in tables[child].collect_schema().names()
            ]
            want = want.join(tables[child], on=keys, how="inner")

        got = view.level(target).collect()
        key = sorted(got.columns)
        assert_frame_equal(got.select(key).sort(key), want.collect().select(key).sort(key))

    @pytest.mark.parametrize("target", LEVELS)
    def test_matches_unpack_of_the_packed_frame(
        self, target: str, view: HierarchyView, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        """The two storage layouts answer the same question identically."""
        want = packer.unpack(packer.pack(flat, ROOT), target)
        got = view.level(target).collect()

        # unpack keeps the nested tail for an intermediate level; level() is flat.
        shared = sorted(set(got.columns) & set(want.columns))
        assert set(got.columns) <= set(want.columns)
        assert_frame_equal(
            got.select(shared).sort(shared),
            want.select(shared).unique().sort(shared),
            check_row_order=False,
        )

    @pytest.mark.parametrize(
        ("target", "expected_rows"),
        [("region", 3), ("store", 5), ("sale", 7)],
    )
    def test_row_count_is_one_per_entity(
        self, target: str, expected_rows: int, view: HierarchyView
    ):
        """The level argument names row granularity, as everywhere else."""
        assert view.level(target).collect().height == expected_rows

    def test_defaults_to_the_finest_level(self, view: HierarchyView):
        assert_frame_equal(
            view.level().collect().sort(SALE_ID),
            view.level("sale").collect().sort(SALE_ID),
        )

    def test_ancestor_attributes_are_in_scope(self, view: HierarchyView):
        """The point of joining the axis: a leaf row can see its region."""
        columns = view.level("sale").collect_schema().names()
        assert REGION_NAME in columns and DISCOUNT in columns

    def test_unknown_level_raises(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.level("planet")


class TestProjectionReachesTheScans:
    """
    The claim the design rests on: an unused ancestor costs its keys, not its data.

    If this stops holding, ``level()`` becomes an unconditional wide join and the
    reason for preferring it over a per-level API disappears — so it is asserted
    against real Parquet scans, where the projection is visible in the plan.
    """

    @staticmethod
    def _scan_block(plan: str, level: str) -> list[str]:
        """
        The plan lines belonging to one level's Parquet scan.

        A scan renders as its SCAN line followed by PROJECT / SELECTION /
        ESTIMATED ROWS, so the block runs from the SCAN to the row estimate.
        """
        lines = [line.strip() for line in plan.splitlines()]
        starts = [i for i, line in enumerate(lines) if line.startswith(f"Parquet SCAN [{level}")]
        assert len(starts) == 1, f"expected exactly one {level} scan in:\n{plan}"
        start = starts[0]
        end = next(i for i in range(start, len(lines)) if lines[i].startswith("ESTIMATED ROWS"))
        return lines[start : end + 1]

    @staticmethod
    def _columns_read(block: list[str]) -> tuple[int, int]:
        """``PROJECT a/b COLUMNS`` as (read, available); ``*`` means all of them."""
        line = next(line for line in block if line.startswith("PROJECT"))
        read, _, available = line.split()[1].partition("/")
        return (int(available) if read == "*" else int(read)), int(available)

    def _plan_for(self, view: HierarchyView, packer: HierarchicalPacker, tmp_path, build):
        view.sink_parquet(tmp_path)
        rescanned = HierarchyView.scan_parquet(tmp_path, packer)
        return build(rescanned).explain().replace(str(tmp_path) + "/", "")

    def test_unused_ancestor_contributes_only_its_key(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """region holds (id, name); wanting no region data should read only id."""
        plan = self._plan_for(
            view, packer, tmp_path, lambda v: v.level("sale").select(SALE_ID, AMOUNT)
        )
        read, available = self._columns_read(self._scan_block(plan, "region"))
        assert available == 2, plan
        assert read == 1, f"expected only the key column to be read:\n{plan}"

    def test_requested_ancestor_column_is_read(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """The complement, so the check above is measuring something."""
        plan = self._plan_for(
            view, packer, tmp_path, lambda v: v.level("sale").select(SALE_ID, REGION_NAME)
        )
        read, available = self._columns_read(self._scan_block(plan, "region"))
        assert read == available == 2, plan

    def test_ancestor_predicate_lands_in_the_ancestor_scan(
        self, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        """A filter on a region attribute runs in region's scan, before the join."""
        plan = self._plan_for(
            view,
            packer,
            tmp_path,
            lambda v: v.level("sale").filter(pl.col(REGION_NAME) == "east"),
        )
        block = self._scan_block(plan, "region")
        assert any(line.startswith("SELECTION") and REGION_NAME in line for line in block), plan
        # ...and nowhere else: the leaf scan should not be carrying it.
        assert not any(
            line.startswith("SELECTION") for line in self._scan_block(plan, "sale")
        ), plan


class TestBranchesDoNotCross:
    """A flat frame has one granularity, so sibling branches are left out."""

    @pytest.fixture
    def branching(self) -> tuple[HierarchyView, HierarchicalPacker]:
        packer = HierarchicalPacker(
            HierarchySpec.from_levels(
                LevelSpec(name="city", id_fields=["id"]),
                LevelSpec(name="street", id_fields=["id"], parent="city", parent_keys=["id"]),
                LevelSpec(name="service", id_fields=["id"], parent="city", parent_keys=["id"]),
            )
        )
        tables = {
            "city": pl.DataFrame({"city.id": [1, 2], "city.pop": [10, 20]}),
            "street": pl.DataFrame(
                {"city.id": [1, 1, 2], "city.street.id": [1, 2, 3], "city.street.len": [5, 6, 7]}
            ),
            "service": pl.DataFrame(
                {"city.id": [1, 2], "city.service.id": [1, 2], "city.service.cost": [50, 60]}
            ),
        }
        return HierarchyView.from_tables(tables, packer), packer

    def test_sibling_branch_is_excluded(self, branching):
        view, _ = branching
        street = view.level("street").collect_schema().names()
        assert "city.service.cost" not in street
        assert "city.pop" in street

    def test_row_count_is_not_the_cross_product(self, branching):
        view, _ = branching
        assert view.level("street").collect().height == 3
        assert view.level("service").collect().height == 2

    def test_omitted_level_is_ambiguous_when_branching(self, branching):
        view, _ = branching
        with pytest.raises(ValueError, match="leaf levels"):
            view.level()
