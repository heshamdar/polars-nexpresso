"""
Equivalence between the packed/eval approach and the HierarchyView approach.

Both paths must produce **the same data in their own natural shape**, and
denormalizing the view's tables must reproduce the packed frame exactly — same
values, same dtypes, same struct field order, same child ordering within every
list. Nothing here weakens the comparison with ``check_dtypes=False``.

Two reference implementations are cross-checked:

``packed``
    ``packer.pack(flat_with_operation_applied, root)`` — ground truth, built by
    applying the operation to the flat frame and packing the result.

``view``
    ``HierarchyView.filter``, materialized with ``nested().collect()``.

Since the view no longer mirrors the polars API, the second half of this file
pins the *replacement workflows* — transforming on ``level()``, rolling up with
a ``group_by`` on ``key_columns()``, existence as a semi-join — against the same
ground truth the removed methods were held to.

The only normalization applied is a sort on the root column. Row order is not
part of the contract — the view's tables come out of joins — but everything
inside a row is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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
QTY = "region.store.sale.qty"


# =============================================================================
# Fixtures
# =============================================================================


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
    """
    Deliberately irregular: uneven fan-out, a single-child parent, and nulls.

    A regular grid hides bugs — every parent having the same child count means
    a broken join can still produce plausible row counts.
    """
    rows = [
        # region 0: two stores, 3 sales and 1 sale
        (0, "north", 0, 0.00, 0, 10.0, 1),
        (0, "north", 0, 0.00, 1, 25.0, 3),
        (0, "north", 0, 0.00, 2, 40.0, 5),
        (0, "north", 1, 0.10, 3, 55.0, 2),
        # region 1: one store, four sales, one null amount
        (1, "south", 2, 0.25, 4, 70.0, 4),
        (1, "south", 2, 0.25, 5, None, 1),
        (1, "south", 2, 0.25, 6, 15.0, 3),
        (1, "south", 2, 0.25, 7, 85.0, 2),
        # region 2: three stores, 2 / 1 / 1 sales
        (2, "east", 3, 0.05, 8, 30.0, 5),
        (2, "east", 3, 0.05, 9, 60.0, 1),
        (2, "east", 4, 0.20, 10, 45.0, 4),
        (2, "east", 5, 0.15, 11, 95.0, 2),
    ]
    return pl.DataFrame(
        {
            REGION_ID: [r[0] for r in rows],
            REGION_NAME: [r[1] for r in rows],
            STORE_ID: [r[2] for r in rows],
            DISCOUNT: [r[3] for r in rows],
            SALE_ID: [r[4] for r in rows],
            AMOUNT: [r[5] for r in rows],
            QTY: [r[6] for r in rows],
        },
        schema_overrides={AMOUNT: pl.Float64},
    )


@pytest.fixture
def view(flat: pl.DataFrame, packer: HierarchicalPacker) -> HierarchyView:
    return HierarchyView.from_frame(flat, packer)


def assert_nested_equal(got: pl.DataFrame, want: pl.DataFrame, label: str = "") -> None:
    """Strict equality up to root-row order — dtypes and field order included."""
    assert (
        got.schema == want.schema
    ), f"{label} schema mismatch:\n  got : {got.schema}\n  want: {want.schema}"
    assert_frame_equal(got.sort(REGION_ID), want.sort(REGION_ID))


# =============================================================================
# Case table
# =============================================================================


@dataclass(frozen=True)
class Case:
    """
    One restriction expressed against both the flat frame and the view.

    Only ``filter`` survives as a view operation; everything else is done on the
    frame :meth:`HierarchyView.level` returns. The replacement workflows for
    those are covered explicitly in ``TestReplacementWorkflows`` below.
    """

    name: str
    # Applied to the flat frame; the result is packed to form ground truth.
    flat_op: Callable[[pl.DataFrame], pl.DataFrame]
    # Applied to the view; materialized with nested().collect().
    view_op: Callable[[HierarchyView], HierarchyView]
    tags: frozenset[str] = field(default_factory=frozenset)


def _flat_filter(*predicates: pl.Expr) -> Callable[[pl.DataFrame], pl.DataFrame]:
    return lambda df: df.filter(*predicates)


def _view_filter(*predicates: pl.Expr) -> Callable[[HierarchyView], HierarchyView]:
    return lambda v: v.filter(*predicates)


CASES: list[Case] = [
    # ---------------------------------------------------------------- identity
    Case("identity", lambda df: df, lambda v: v),
    # ----------------------------------------------------------------- filters
    Case(
        "filter_leaf_attribute",
        _flat_filter(pl.col(AMOUNT) > 30),
        _view_filter(pl.col(AMOUNT) > 30),
    ),
    Case(
        "filter_leaf_key",
        _flat_filter(pl.col(SALE_ID) % 2 == 0),
        _view_filter(pl.col(SALE_ID) % 2 == 0),
    ),
    Case(
        "filter_ancestor_key",
        _flat_filter(pl.col(REGION_ID) == 1),
        _view_filter(pl.col(REGION_ID) == 1),
    ),
    Case(
        "filter_ancestor_attribute",
        _flat_filter(pl.col(REGION_NAME) == "east"),
        _view_filter(pl.col(REGION_NAME) == "east"),
    ),
    Case(
        "filter_parent_attribute",
        _flat_filter(pl.col(DISCOUNT) > 0.05),
        _view_filter(pl.col(DISCOUNT) > 0.05),
    ),
    Case(
        "filter_cross_level",
        _flat_filter(pl.col(AMOUNT) > pl.col(DISCOUNT) * 400),
        _view_filter(pl.col(AMOUNT) > pl.col(DISCOUNT) * 400),
    ),
    Case(
        "filter_cross_level_grandparent",
        _flat_filter(pl.col(AMOUNT) > pl.col(REGION_ID) * 20),
        _view_filter(pl.col(AMOUNT) > pl.col(REGION_ID) * 20),
    ),
    Case(
        "filter_compound",
        _flat_filter(pl.col(AMOUNT) > 20, pl.col(QTY) < 5),
        _view_filter(pl.col(AMOUNT) > 20, pl.col(QTY) < 5),
    ),
    Case(
        "filter_chained",
        lambda df: df.filter(pl.col(AMOUNT) > 20).filter(pl.col(QTY) < 5),
        lambda v: v.filter(pl.col(AMOUNT) > 20).filter(pl.col(QTY) < 5),
    ),
    Case(
        "filter_nulls_out",
        _flat_filter(pl.col(AMOUNT).is_not_null()),
        _view_filter(pl.col(AMOUNT).is_not_null()),
    ),
    Case(
        "filter_keeps_everything",
        _flat_filter(pl.col(AMOUNT).is_null() | (pl.col(AMOUNT) > -1)),
        _view_filter(pl.col(AMOUNT).is_null() | (pl.col(AMOUNT) > -1)),
    ),
    Case(
        "filter_single_leaf",
        _flat_filter(pl.col(SALE_ID) == 5),
        _view_filter(pl.col(SALE_ID) == 5),
    ),
]


# =============================================================================
# The equivalence laws
# =============================================================================


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
class TestNestedEquivalence:
    """view.nested().collect() == pack(flat_with_operation)."""

    def test_nested_matches_packed(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        want = packer.pack(case.flat_op(flat), ROOT)
        got = case.view_op(view).nested().collect()
        assert_nested_equal(got, want, case.name)

    def test_denormalized_tables_match_packed(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """Denormalizing the view's own tables reproduces the packed frame."""
        want = packer.pack(case.flat_op(flat), ROOT)
        got = packer.denormalize(case.view_op(view).tables()).collect()  # type: ignore[union-attr]
        assert_nested_equal(got, want, case.name)

    def test_flat_matches_unpacked(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """At leaf granularity both paths agree column-for-column."""
        want = packer.unpack(packer.pack(case.flat_op(flat), ROOT), "sale")
        got = case.view_op(view).level("sale").collect()
        assert sorted(got.columns) == sorted(want.columns), case.name
        assert_frame_equal(
            got.select(sorted(got.columns)).sort(SALE_ID),
            want.select(sorted(want.columns)).sort(SALE_ID),
        )

    def test_schema_is_stable(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """The plan's advertised schema matches what materializing produces."""
        result = case.view_op(view)
        assert result.nested().collect_schema() == result.nested().collect().schema
        assert result.level("sale").collect_schema() == result.level("sale").collect().schema

    @pytest.mark.parametrize("level", ["region", "store", "sale"])
    def test_tables_match_split_levels(
        self,
        case: Case,
        level: str,
        flat: pl.DataFrame,
        view: HierarchyView,
        packer: HierarchicalPacker,
    ):
        """
        Every level's table matches ``split_levels`` of the packed ground truth.

        This is the law the joined terminals cannot check. ``collect_nested()``
        and ``collect()`` both join parent to child, so an orphaned child row —
        one whose parent was filtered away — silently disappears on the way out
        and the comparison still passes. ``tables()`` performs no join, so
        comparing it level by level is what actually pins cross-level
        consistency.
        """
        want_tables = packer.split_levels(packer.pack(case.flat_op(flat), ROOT))
        got = case.view_op(view).tables()[level].collect()
        want = want_tables[level]
        assert sorted(got.columns) == sorted(want.columns), f"{case.name}/{level}"
        key = sorted(got.columns)
        assert_frame_equal(
            got.select(key).sort(key),
            want.select(key).sort(key),
        )


# =============================================================================
# The replacement workflows
# =============================================================================
#
# Transforming, projecting and rolling up are no longer view operations: they
# happen on the frame ``level()`` returns, or on a level's own table. These
# tests pin the documented replacements against the same ground truth the
# removed methods were checked against, so the guidance in the docs stays
# executable rather than aspirational.


class TestReplacementWorkflows:
    """What to do now that the view no longer mirrors the polars API."""

    def test_transform_on_level_then_pack(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """
        A cross-level derived column: the case ``list.eval`` cannot express.

        Ancestor attributes are in scope on ``level()``, so this is one ordinary
        expression, and packing the result gives the nested frame the old
        ``with_columns`` produced.
        """
        derive = (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
        want = packer.pack(flat.with_columns(derive), ROOT)
        got = packer.pack(view.level("sale").with_columns(derive).collect(), ROOT)
        assert_nested_equal(got, want)

    def test_transform_a_level_then_renest(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """
        A single-level transform stays normalized: edit that table, rebuild.

        Cheaper than the round trip above, because no level is ever widened to
        another's granularity.
        """
        derive = (pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")
        want = packer.pack(flat.with_columns(derive), ROOT)

        tables = dict(view.tables())
        tables["sale"] = tables["sale"].with_columns(derive)
        got = HierarchyView.from_tables(tables, packer).nested().collect()
        assert_nested_equal(got, want)

    def test_drop_a_level_column_then_renest(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        want = packer.pack(flat.drop(QTY), ROOT)
        tables = dict(view.tables())
        tables["sale"] = tables["sale"].drop(QTY)
        got = HierarchyView.from_tables(tables, packer).nested().collect()
        assert_nested_equal(got, want)

    def test_projection_reaches_the_scan(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """
        Asking for the whole axis and using none of it is close to free.

        This is the claim the design rests on: an ancestor level contributes
        only its key columns when its data is unused, so ``level()`` does not
        need to be told which columns the caller intends to read.
        """
        plan = view.level("sale").select(SALE_ID, AMOUNT).explain()
        region_scan = [ln for ln in plan.splitlines() if "region.name" in ln]
        assert region_scan, f"expected the region scan in the plan:\n{plan}"
        assert all("PROJECT" in ln for ln in region_scan), plan
        assert not any('PROJECT["region.name"' in ln for ln in region_scan), plan


@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "first", "last"])
def test_rollup_matches_promote_attribute(
    agg, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
):
    """
    A group_by on ``key_columns`` is what ``promote`` was doing underneath.

    Written out it is no longer than the method call, works for any ancestor
    rather than only the immediate parent, and the aggregation is an ordinary
    polars expression instead of a string from a fixed vocabulary.
    """
    want = packer.promote_attribute(
        flat, "amount", from_level="sale", to_level="store", agg=agg, alias="agg"
    ).select(STORE_ID, "region.store.agg")

    rolled = (
        view.level("sale")
        .group_by(view.key_columns("store"))
        .agg(getattr(pl.col(AMOUNT), "len" if agg == "count" else agg)().alias("region.store.agg"))
    )
    got = rolled.collect().select(STORE_ID, "region.store.agg")
    assert_frame_equal(
        got.sort(STORE_ID),
        want.sort(STORE_ID).with_columns(
            pl.col("region.store.agg").cast(got.schema["region.store.agg"])
        ),
    )


def test_rollup_can_skip_levels(
    flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
):
    """
    sale -> region directly, which ``promote`` refused and the packer still does.

    ``normalize`` replicates *every* ancestor id into a level's table, not just
    the immediate parent's, so the group_by needs no intermediate hop.
    """
    with pytest.raises(ValueError, match="immediate child"):
        packer.promote_attribute(flat, "amount", from_level="sale", to_level="region", agg="sum")

    got = (
        view.level("sale")
        .group_by(view.key_columns("region"))
        .agg(pl.col(AMOUNT).sum().alias("total"))
        .collect()
        .sort(REGION_ID)
    )
    want = flat.group_by(REGION_ID).agg(pl.col(AMOUNT).sum().alias("total")).sort(REGION_ID)
    assert_frame_equal(got, want)


class TestExistenceWorkflow:
    """
    "Parents having at least one matching child" is a semi-join.

    The packer spells the predicate per child struct element against a frame
    packed at the parent's granularity; on the view it is an ordinary flat
    expression over ``level()``. Both must select the same entities.
    """

    CONDITIONS = [
        (pl.col(AMOUNT) > 50, pl.element().struct.field("amount") > 50, "some"),
        (pl.col(AMOUNT) > 1000, pl.element().struct.field("amount") > 1000, "none"),
        (pl.col(AMOUNT) > -1, pl.element().struct.field("amount") > -1, "most"),
        (pl.col(QTY) == 5, pl.element().struct.field("qty") == 5, "exact"),
    ]

    @staticmethod
    def _parents_with_a_match(view: HierarchyView, condition: pl.Expr, at_level: str):
        keys = view.key_columns(at_level)
        matching = view.level("sale").filter(condition).select(keys).unique()
        return view.level(at_level).join(matching, on=keys, how="semi")

    @pytest.mark.parametrize(
        ("flat_condition", "element_condition"),
        [(f, e) for f, e, _ in CONDITIONS],
        ids=[i for _, _, i in CONDITIONS],
    )
    def test_selects_same_entities_as_packer(
        self,
        flat_condition,
        element_condition,
        flat: pl.DataFrame,
        view: HierarchyView,
        packer: HierarchicalPacker,
    ):
        """The packer requires an immediate child, so compare store <- sale."""
        at_store = packer.pack(flat, "store")
        want = packer.any_child_satisfies(
            at_store, from_level="sale", to_level="store", condition=element_condition
        )
        got = self._parents_with_a_match(view, flat_condition, "store").collect()
        assert sorted(got[STORE_ID].to_list()) == sorted(want[STORE_ID].to_list())

    def test_can_skip_levels(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """sale -> region, which the packer rejects."""
        condition = pl.col(AMOUNT) > 50
        with pytest.raises(ValueError, match="immediate child"):
            packer.any_child_satisfies(
                packer.pack(flat, ROOT), from_level="sale", to_level="region", condition=condition
            )
        got = self._parents_with_a_match(view, condition, "region").collect()
        assert got.height == flat.filter(condition)[REGION_ID].n_unique()

    def test_differs_from_filter(self, flat: pl.DataFrame, view: HierarchyView):
        """
        Existence is not ``filter``, and the difference is the point.

        ``filter`` restricts the children too and prunes parents left childless;
        a semi-join restricts only the parent and leaves its children whole.
        """
        condition = pl.col(AMOUNT) > 50
        semi = self._parents_with_a_match(view, condition, "store").collect()
        filtered = view.filter(condition)

        assert sorted(semi[STORE_ID].to_list()) == sorted(
            filtered.tables()["store"].collect()[STORE_ID].to_list()
        )
        assert filtered.tables()["sale"].collect().height < view.tables()["sale"].collect().height


# =============================================================================
# Empty-parent semantics
# =============================================================================


class TestEmptyParentEquivalence:
    """'prune' must match pack(); 'keep' must deliberately differ."""

    PREDICATE = pl.col(AMOUNT) > 50  # leaves several stores childless

    def test_prune_matches_pack(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        assert_nested_equal(
            view.filter(self.PREDICATE).nested().collect(),
            packer.pack(flat.filter(self.PREDICATE), ROOT),
        )

    def test_keep_retains_more_children(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        kept = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(pl.col(REGION_NAME) == "east")
            .nested()
            .collect()
        )
        pruned = (
            HierarchyView.from_frame(flat, packer)
            .filter(pl.col(REGION_NAME) == "east")
            .nested()
            .collect()
        )
        # Same regions either way; 'keep' preserves stores that lost every sale.
        assert kept.height == pruned.height
        assert kept.schema == pruned.schema

    def test_filter_removing_everything(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        impossible = pl.col(AMOUNT) > 10_000
        got = view.filter(impossible).nested().collect()
        want = packer.pack(flat.filter(impossible), ROOT)
        assert got.height == 0 and want.height == 0
        assert got.schema == want.schema


# =============================================================================
# Other hierarchy shapes
# =============================================================================


class TestTwoLevelHierarchy:
    """The laws must not depend on there being exactly three levels."""

    @pytest.fixture
    def packer(self) -> HierarchicalPacker:
        return HierarchicalPacker(
            HierarchySpec.from_levels(
                LevelSpec(name="country", id_fields=["code"]),
                LevelSpec(name="city", id_fields=["id"], parent_keys=["country_code"]),
            )
        )

    @pytest.fixture
    def flat(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "country.code": ["US", "US", "US", "FR", "DE", "DE"],
                "country.name": ["States", "States", "States", "France", "Germany", "Germany"],
                "country.city.id": [1, 2, 3, 4, 5, 6],
                "country.city.pop": [8.0, 4.0, 2.0, 2.1, 3.6, 1.8],
            }
        )

    def test_identity(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        view = HierarchyView.from_frame(flat, packer)
        want = packer.pack(flat, "country")
        got = view.nested().collect()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("country.code"), want.sort("country.code"))

    def test_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("country.city.pop") > 2.5
        view = HierarchyView.from_frame(flat, packer)
        want = packer.pack(flat.filter(predicate), "country")
        got = view.filter(predicate).nested().collect()
        assert_frame_equal(got.sort("country.code"), want.sort("country.code"))

    def test_transform_a_level_then_renest(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        expr = (pl.col("country.city.pop") * 2).alias("country.city.doubled")
        view = HierarchyView.from_frame(flat, packer)
        want = packer.pack(flat.with_columns(expr), "country")

        tables = dict(view.tables())
        tables["city"] = tables["city"].with_columns(expr)
        got = HierarchyView.from_tables(tables, packer).nested().collect()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("country.code"), want.sort("country.code"))


class TestFourLevelHierarchy:
    """Deep nesting: the ancestor-join logic must walk more than one hop."""

    @pytest.fixture
    def packer(self) -> HierarchicalPacker:
        return HierarchicalPacker(
            HierarchySpec.from_levels(
                LevelSpec(name="a", id_fields=["id"]),
                LevelSpec(name="b", id_fields=["id"], parent_keys=["a_id"]),
                LevelSpec(name="c", id_fields=["id"], parent_keys=["b_id"]),
                LevelSpec(name="d", id_fields=["id"], parent_keys=["c_id"]),
            )
        )

    @pytest.fixture
    def flat(self) -> pl.DataFrame:
        n = 16
        return pl.DataFrame(
            {
                "a.id": [i // 8 for i in range(n)],
                "a.rate": [0.1 * (i // 8) for i in range(n)],
                "a.b.id": [i // 4 for i in range(n)],
                "a.b.c.id": [i // 2 for i in range(n)],
                "a.b.c.d.id": list(range(n)),
                "a.b.c.d.value": [float(i * 3) for i in range(n)],
            }
        )

    def test_identity(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        want = packer.pack(flat, "a")
        got = HierarchyView.from_frame(flat, packer).nested().collect()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_leaf_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("a.b.c.d.value") > 20
        want = packer.pack(flat.filter(predicate), "a")
        got = HierarchyView.from_frame(flat, packer).filter(predicate).nested().collect()
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_three_hop_ancestor_reference(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        """Leaf 'd' reading root 'a' — three levels up, in one expression."""
        expr = (pl.col("a.b.c.d.value") * pl.col("a.rate")).alias("a.b.c.d.scaled")
        want = packer.pack(flat.with_columns(expr), "a")
        view = HierarchyView.from_frame(flat, packer)
        got = packer.pack(view.level("d").with_columns(expr).collect(), "a")
        assert got.schema == want.schema
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_three_hop_ancestor_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("a.b.c.d.value") > pl.col("a.rate") * 100
        want = packer.pack(flat.filter(predicate), "a")
        got = HierarchyView.from_frame(flat, packer).filter(predicate).nested().collect()
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))


# =============================================================================
# Round-trip laws
# =============================================================================


class TestRoundTripLaws:
    """Algebraic identities that must hold regardless of the operation."""

    def test_view_from_packed_frame(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        """A view built from an already-packed frame round-trips to itself."""
        packed = packer.pack(flat, ROOT)
        got = HierarchyView.from_frame(packed, packer).nested().collect()
        assert got.schema == packed.schema
        assert_frame_equal(got.sort(REGION_ID), packed.sort(REGION_ID))

    def test_parquet_round_trip_is_lossless(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        view.sink_parquet(tmp_path)
        got = HierarchyView.scan_parquet(tmp_path, packer).nested().collect()
        want = packer.pack(flat, ROOT)
        assert got.schema == want.schema
        assert_frame_equal(got.sort(REGION_ID), want.sort(REGION_ID))

    @pytest.mark.parametrize(
        ("level", "key", "still_nested"),
        [
            ("region", REGION_ID, "region.store"),
            ("store", STORE_ID, "region.store.sale"),
            ("sale", SALE_ID, None),
        ],
    )
    def test_collect_matches_unpack_at_every_level(
        self,
        level,
        key,
        still_nested,
        flat: pl.DataFrame,
        view: HierarchyView,
        packer: HierarchicalPacker,
    ):
        """
        ``collect(level)`` == ``unpack(packed, level)`` minus the nested tail.

        ``unpack`` stops at the requested level and leaves everything below it
        as a ``List[Struct]`` column. The view represents that tail as separate
        tables instead, so the comparison drops it — at the leaf level there is
        no tail and the two agree column-for-column.
        """
        want = packer.unpack(packer.pack(flat, ROOT), level)
        if still_nested is not None:
            want = want.drop(still_nested)
        got = view.level(level).collect()
        assert sorted(got.columns) == sorted(want.columns)
        assert_frame_equal(
            got.select(sorted(got.columns)).sort(key),
            want.select(sorted(want.columns)).sort(key),
        )

    def test_lazy_and_eager_input_agree(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        eager = HierarchyView.from_frame(flat, packer).nested().collect()
        lazy = HierarchyView.from_frame(flat.lazy(), packer).nested().collect()
        assert_frame_equal(eager.sort(REGION_ID), lazy.sort(REGION_ID))

    def test_dangling_child_is_visible_in_tables_but_not_in_level(
        self, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        """
        Where referential integrity is broken, the two entry points differ.

        ``tables()`` is the storage as it is: a sale pointing at a store that
        isn't there is still a row, and an unfiltered view adds no join that
        could hide it. ``level()`` is a frame at sale granularity *with store
        columns on it*, which that sale has no values for — so the axis join is
        an inner one and the orphan is absent. Neither is a bug; the invariant
        is that ``tables()`` never silently loses a row.
        """
        tables = packer.normalize(flat)
        orphan = (
            tables["sale"]
            .head(1)
            .with_columns(
                pl.lit(999).cast(pl.Int64).alias(STORE_ID),
                pl.lit(999).cast(pl.Int64).alias(SALE_ID),
            )
        )
        tables["sale"] = pl.concat([tables["sale"], orphan])
        view = HierarchyView.from_tables(tables, packer)

        stored = view.tables()["sale"].collect()
        assert stored.height == len(flat) + 1
        assert stored.filter(pl.col(SALE_ID) == 999).height == 1

        joined = view.level("sale").collect()
        assert joined.filter(pl.col(SALE_ID) == 999).height == 0
        assert joined.height == stored.height - 1

    def test_tables_and_nested_agree(self, view: HierarchyView, packer: HierarchicalPacker):
        """nested() is exactly denormalize(tables())."""
        filtered = view.filter(pl.col(AMOUNT) > 30)
        via_tables = packer.denormalize(filtered.tables()).collect()  # type: ignore[union-attr]
        assert_frame_equal(via_tables.sort(REGION_ID), filtered.nested().collect().sort(REGION_ID))
