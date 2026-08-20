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


class TestWithLevel:
    """
    ``with_level`` is the counterpart to ``level``: it keeps the hierarchy.

    ``level(g)`` hands you a frame and lets go, which is what you want for a
    query. When the result should still be filterable, nestable or sinkable, the
    modification has to land back on a level's table instead.
    """

    DOUBLED = "region.store.sale.doubled"

    def _double(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_columns((pl.col(AMOUNT) * 2).alias(self.DOUBLED))

    def test_returns_a_view_that_still_composes(self, view: HierarchyView):
        widened = view.with_level("sale", self._double)
        assert isinstance(widened, HierarchyView)
        assert self.DOUBLED in widened.level("sale").collect_schema().names()
        # ...and the hierarchy is intact, so the rest of the API still applies.
        assert widened.filter(pl.col(self.DOUBLED) > 100).level("sale").collect().height > 0

    def test_the_column_reaches_the_nested_shape(self, view: HierarchyView):
        nested = view.with_level("sale", self._double).nested().collect()
        store = nested.schema["region.store"].inner.to_schema()  # type: ignore[union-attr]
        assert "doubled" in store["sale"].inner.to_schema()

    def test_preserves_empty_parents(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        """The tables()/from_tables round trip silently resets this; with_level must not."""
        keep = HierarchyView.from_frame(flat, packer, empty_parents="keep")
        assert "keep" in repr(keep.with_level("sale", self._double))

    def test_rejects_an_unqualified_name(self, view: HierarchyView):
        """
        The footgun this guard exists for.

        An unqualified column survives level() and is silently dropped by
        nested(), so the loss shows up far from its cause.
        """
        with pytest.raises(ValueError, match="name no level in this view"):
            view.with_level("sale", lambda lf: lf.with_columns(pl.col(AMOUNT).alias("doubled")))

    def test_rejects_a_foreign_level_name(self, view: HierarchyView):
        """Landing a column on another level is possible, but never by accident."""
        with pytest.raises(ValueError, match="named for another level"):
            view.with_level("sale", lambda lf: lf.with_columns(pl.lit(1).alias("region.oops")))

    def test_rejects_dropping_a_key(self, view: HierarchyView):
        with pytest.raises(ValueError, match="key column"):
            view.with_level("sale", lambda lf: lf.drop(REGION_ID))

    def test_rejects_unknown_level(self, view: HierarchyView):
        with pytest.raises(KeyError, match="not present in this view"):
            view.with_level("planet", self._double)

    def test_matches_the_manual_round_trip(self, view: HierarchyView, packer: HierarchicalPacker):
        tables = dict(view.tables())
        tables["sale"] = self._double(tables["sale"])
        manual = HierarchyView.from_tables(tables, packer).nested().collect()
        assert_frame_equal(
            view.with_level("sale", self._double).nested().collect().sort(REGION_ID),
            manual.sort(REGION_ID),
        )


class TestWithLevelSeesAncestorAttributes:
    """
    A transform may reference any ancestor attribute, not just this level's own
    columns and the ancestor *keys* that ``normalize`` replicates. The columns
    are joined in for the computation and dropped again, so the level keeps its
    own schema and ``nested()`` still places everything by path.
    """

    NET = "region.store.sale.net"

    def _price(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Needs DISCOUNT, which lives on ``store``, not ``sale``."""
        return lf.with_columns((pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias(self.NET))

    def test_cross_level_derivation_needs_no_manual_join(self, view: HierarchyView):
        priced = view.with_level("sale", self._price)
        assert self.NET in priced.tables()["sale"].collect_schema().names()

    def test_the_values_match_the_flat_computation(self, view: HierarchyView, flat: pl.DataFrame):
        got = (
            view.with_level("sale", self._price)
            .level("sale")
            .collect()
            .sort(SALE_ID)
            .get_column(self.NET)
            .to_list()
        )
        want = (
            flat.sort(SALE_ID)
            .with_columns((pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias(self.NET))
            .get_column(self.NET)
            .to_list()
        )
        assert got == want

    def test_borrowed_columns_are_lent_not_adopted(self, view: HierarchyView):
        """The ancestor attribute must not linger on the level's own table."""
        schema = view.with_level("sale", self._price).tables()["sale"].collect_schema().names()
        assert DISCOUNT not in schema
        assert REGION_NAME not in schema

    def test_borrowing_does_not_change_the_row_count(self, view: HierarchyView):
        """The join is LEFT on purpose: borrowing must never drop a row."""
        before = view.tables()["sale"].collect().height
        assert view.with_level("sale", self._price).tables()["sale"].collect().height == before

    def test_an_ancestor_value_can_be_kept_by_naming_it_for_this_level(self, view: HierarchyView):
        """Aliasing to this level's path makes a copy that survives the drop."""
        snapshot = "region.store.sale.discount_at_sale"
        kept = view.with_level("sale", lambda lf: lf.with_columns(pl.col(DISCOUNT).alias(snapshot)))
        assert snapshot in kept.tables()["sale"].collect_schema().names()
        assert DISCOUNT not in kept.tables()["sale"].collect_schema().names()

    def test_the_derived_column_reaches_the_nested_shape(self, view: HierarchyView):
        nested = view.with_level("sale", self._price).nested().collect()
        store = nested.schema["region.store"].inner.to_schema()  # type: ignore[union-attr]
        assert "net" in store["sale"].inner.to_schema()

    def test_a_descendant_column_is_still_refused(self, view: HierarchyView):
        """Borrowing goes up the hierarchy only; pulling a child down would fan out."""
        with pytest.raises(ValueError, match="not an ancestor"):
            view.with_level("region", lambda lf: lf.with_columns(pl.col(AMOUNT).alias("region.x")))

    def test_an_unknown_column_is_named_rather_than_dumping_the_plan(self, view: HierarchyView):
        with pytest.raises(KeyError, match="Unknown column"):
            view.with_level(
                "sale", lambda lf: lf.with_columns(pl.col("region.nope").alias(self.NET))
            )

    def test_a_same_level_transform_joins_nothing(self, view: HierarchyView):
        """
        The widening is only paid when the transform reaches for an ancestor.

        A transform that names none is run against the level's own table, so the
        common case costs no join at all.
        """
        plan = (
            view.with_level("sale", lambda lf: lf.with_columns(pl.col(AMOUNT).alias(self.NET)))
            .tables()["sale"]
            .explain()
        )
        assert "JOIN" not in plan.upper()

    def test_one_join_per_ancestor_level_not_per_column(self, view: HierarchyView):
        """Two attributes from the same ancestor share a single join."""
        plan = (
            view.with_level(
                "sale",
                lambda lf: lf.with_columns(
                    (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias(self.NET),
                    pl.col(REGION_NAME).alias("region.store.sale.region_name"),
                ),
            )
            .tables()["sale"]
            .explain()
        )
        # region and store contribute one attribute each here; explain() prints
        # an opening and a closing marker per join.
        assert plan.count("LEFT JOIN:") == 2


class TestPromoteToAnAncestor:
    """
    A transform names its output for a level, and that is where the column goes.

    Working at ``sale`` granularity there are many sale rows per region, so a
    column named ``region.x`` has many values per region row and something must
    reduce them. ``promote`` says how — and by defaulting to ``None``, that a
    column landing on another level is never implicit.
    """

    REVENUE = "region.revenue"
    BEST = "region.store.best"

    def _revenue(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """A window sum: constant within each region by construction."""
        return lf.with_columns(pl.col(AMOUNT).sum().over(REGION_ID).alias(self.REVENUE))

    def test_refused_without_promote(self, view: HierarchyView):
        with pytest.raises(ValueError, match="named for another level"):
            view.with_level("sale", self._revenue)

    def test_the_refusal_names_both_modes(self, view: HierarchyView):
        """The error has to teach the fix, since the column name alone looks fine."""
        with pytest.raises(ValueError, match="promote='first'.*promote='list'"):
            view.with_level("sale", self._revenue)

    def test_first_lands_the_rollup_on_the_ancestor(self, view: HierarchyView, flat: pl.DataFrame):
        promoted = view.with_level("sale", self._revenue, promote="first")
        got = promoted.tables()["region"].collect().sort(REGION_ID)
        want = flat.group_by(REGION_ID).agg(pl.col(AMOUNT).sum()).sort(REGION_ID)
        assert got[self.REVENUE].to_list() == want[AMOUNT].to_list()

    def test_the_column_leaves_the_originating_level(self, view: HierarchyView):
        promoted = view.with_level("sale", self._revenue, promote="first")
        assert self.REVENUE not in promoted.tables()["sale"].collect_schema().names()
        assert self.REVENUE in promoted.tables()["region"].collect_schema().names()

    def test_promoting_adds_no_rows_to_the_ancestor(self, view: HierarchyView):
        """It is a left join on the ancestor's own keys, so the row set is untouched."""
        before = view.tables()["region"].collect().height
        promoted = view.with_level("sale", self._revenue, promote="first")
        assert promoted.tables()["region"].collect().height == before

    def test_list_gathers_every_child_value(self, view: HierarchyView, flat: pl.DataFrame):
        gathered = view.with_level(
            "sale",
            lambda lf: lf.with_columns(pl.col(AMOUNT).alias("region.amounts")),
            promote="list",
        )
        got = gathered.tables()["region"].collect().sort(REGION_ID)
        assert got.schema["region.amounts"] == pl.List(pl.Float64)
        want = flat.group_by(REGION_ID).agg(pl.col(AMOUNT)).sort(REGION_ID)
        assert [sorted(v) for v in got["region.amounts"]] == [sorted(v) for v in want[AMOUNT]]

    def test_several_levels_in_one_pass(self, view: HierarchyView):
        both = view.with_level(
            "sale",
            lambda lf: lf.with_columns(
                pl.col(AMOUNT).sum().over(REGION_ID).alias(self.REVENUE),
                pl.col(AMOUNT).max().over(STORE_ID).alias(self.BEST),
                (pl.col(AMOUNT) * 2).alias("region.store.sale.dbl"),
            ),
            promote="first",
        )
        assert self.REVENUE in both.tables()["region"].collect_schema().names()
        assert self.BEST in both.tables()["store"].collect_schema().names()
        assert "region.store.sale.dbl" in both.tables()["sale"].collect_schema().names()

    def test_the_promoted_column_is_first_class(self, view: HierarchyView):
        """It is stored on the ancestor, so filter routes and cascades from there."""
        promoted = view.with_level("sale", self._revenue, promote="first")
        kept = promoted.filter(pl.col(self.REVENUE) > 0)
        assert kept.tables()["region"].collect().height > 0
        assert self.REVENUE in promoted.nested().collect().columns

    def test_a_descendant_is_still_refused(self, view: HierarchyView):
        """Reducing onto a coarser row is defined; fanning out to a finer one is not."""
        with pytest.raises(ValueError, match="a descendant of"):
            view.with_level(
                "region", lambda lf: lf.with_columns(pl.lit(1).alias(self.BEST)), promote="first"
            )

    def test_an_unqualified_name_is_still_refused(self, view: HierarchyView):
        with pytest.raises(ValueError, match="name no level in this view"):
            view.with_level(
                "sale", lambda lf: lf.with_columns(pl.col(AMOUNT).alias("oops")), promote="first"
            )

    def test_colliding_with_a_stored_ancestor_column_is_refused(self, view: HierarchyView):
        """Otherwise a borrowed column silently overwrites the one it was lent from."""
        with pytest.raises(ValueError, match="already has"):
            view.with_level(
                "sale",
                lambda lf: lf.with_columns(pl.lit("x").alias(REGION_NAME)),
                promote="first",
            )

    def test_an_unknown_mode_is_rejected(self, view: HierarchyView):
        with pytest.raises(ValueError, match="Invalid promote"):
            view.with_level("sale", self._revenue, promote="mean")  # type: ignore[arg-type]

    def test_first_takes_the_caller_at_their_word(self, view: HierarchyView):
        """
        Nothing verifies uniformity — that is the documented bargain.

        A varying column keeps one arbitrary value rather than raising, which is
        why the mode has to be asked for by name.
        """
        loose = view.with_level(
            "sale",
            lambda lf: lf.with_columns((pl.col(AMOUNT) * 2).alias("region.arbitrary")),
            promote="first",
        )
        values = loose.tables()["region"].collect()["region.arbitrary"].to_list()
        assert len(values) == loose.tables()["region"].collect().height
        assert all(v is not None for v in values)
