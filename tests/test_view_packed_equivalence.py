"""
Equivalence between the packed/eval approach and the HierarchyView approach.

Both paths must produce **the same data in their own natural shape**, and
denormalizing the view's tables must reproduce the packed frame exactly — same
values, same dtypes, same struct field order, same child ordering within every
list. Nothing here weakens the comparison with ``check_dtypes=False``.

Three reference implementations are cross-checked:

``packed``
    ``packer.pack(flat_with_operation_applied, root)`` — ground truth, built by
    applying the operation to the flat frame and packing the result.

``eval``
    ``apply_nested_operations`` over an already-packed frame — nexpresso's
    nested expression builder, i.e. the ``list.eval`` route. Only used for
    operations that route can actually express.

``view``
    ``HierarchyView`` operations, materialized with ``collect_nested()``.

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

from nexpresso import (
    HierarchicalPacker,
    HierarchySpec,
    HierarchyView,
    LevelSpec,
    apply_nested_operations,
)

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
    """One operation expressed against both the flat frame and the view."""

    name: str
    # Applied to the flat frame; the result is packed to form ground truth.
    flat_op: Callable[[pl.DataFrame], pl.DataFrame]
    # Applied to the view; materialized with collect_nested().
    view_op: Callable[[HierarchyView], HierarchyView]
    # Optional equivalent expressed with the nested expression builder.
    eval_fields: dict | None = None
    eval_mode: str = "with_fields"
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
    # -------------------------------------------------------------- transforms
    Case(
        "transform_leaf_in_place",
        lambda df: df.with_columns((pl.col(AMOUNT) * 2).alias(AMOUNT)),
        lambda v: v.with_columns((pl.col(AMOUNT) * 2).alias(AMOUNT)),
        eval_fields={"region.store": {"sale": {"amount": lambda x: x * 2}}},
    ),
    Case(
        "transform_leaf_new_column",
        lambda df: df.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")),
        lambda v: v.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")),
    ),
    Case(
        "transform_parent_in_place",
        lambda df: df.with_columns((pl.col(DISCOUNT) * 100).alias(DISCOUNT)),
        lambda v: v.with_columns((pl.col(DISCOUNT) * 100).alias(DISCOUNT)),
        eval_fields={"region.store": {"discount": lambda x: x * 100}},
    ),
    Case(
        "transform_root_in_place",
        lambda df: df.with_columns(pl.col(REGION_NAME).str.to_uppercase().alias(REGION_NAME)),
        lambda v: v.with_columns(pl.col(REGION_NAME).str.to_uppercase().alias(REGION_NAME)),
        eval_fields={"region.name": lambda x: x.str.to_uppercase()},
    ),
    Case(
        "transform_cross_level_parent",
        lambda df: df.with_columns(
            (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
        ),
        lambda v: v.with_columns(
            (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
        ),
    ),
    Case(
        "transform_cross_level_grandparent",
        lambda df: df.with_columns(
            (pl.col(AMOUNT) + pl.col(REGION_ID)).alias("region.store.sale.adj")
        ),
        lambda v: v.with_columns(
            (pl.col(AMOUNT) + pl.col(REGION_ID)).alias("region.store.sale.adj")
        ),
    ),
    Case(
        "transform_null_propagation",
        lambda df: df.with_columns((pl.col(AMOUNT) + 1).alias("region.store.sale.plus")),
        lambda v: v.with_columns((pl.col(AMOUNT) + 1).alias("region.store.sale.plus")),
    ),
    Case(
        "transform_null_fill",
        lambda df: df.with_columns(pl.col(AMOUNT).fill_null(0.0).alias(AMOUNT)),
        lambda v: v.with_columns(pl.col(AMOUNT).fill_null(0.0).alias(AMOUNT)),
        eval_fields={"region.store": {"sale": {"amount": lambda x: x.fill_null(0.0)}}},
    ),
    Case(
        "transform_conditional",
        lambda df: df.with_columns(
            pl.when(pl.col(QTY) >= 3)
            .then(pl.col(AMOUNT))
            .otherwise(0.0)
            .alias("region.store.sale.bulk")
        ),
        lambda v: v.with_columns(
            pl.when(pl.col(QTY) >= 3)
            .then(pl.col(AMOUNT))
            .otherwise(0.0)
            .alias("region.store.sale.bulk")
        ),
    ),
    Case(
        "transform_window_over_parent",
        lambda df: df.with_columns(
            pl.col(AMOUNT).sum().over(STORE_ID).alias("region.store.sale.store_total")
        ),
        lambda v: v.with_columns(
            pl.col(AMOUNT).sum().over(STORE_ID).alias("region.store.sale.store_total")
        ),
    ),
    Case(
        "transform_multiple_columns",
        lambda df: df.with_columns(
            (pl.col(AMOUNT) * 2).alias("region.store.sale.a"),
            (pl.col(QTY) + 1).alias("region.store.sale.b"),
        ),
        lambda v: v.with_columns(
            (pl.col(AMOUNT) * 2).alias("region.store.sale.a"),
            (pl.col(QTY) + 1).alias("region.store.sale.b"),
        ),
    ),
    # ------------------------------------------------------------------- drops
    Case(
        "drop_leaf_attribute",
        lambda df: df.drop(QTY),
        lambda v: v.drop(QTY),
    ),
    Case(
        "drop_parent_attribute",
        lambda df: df.drop(DISCOUNT),
        lambda v: v.drop(DISCOUNT),
    ),
    Case(
        "drop_root_attribute",
        lambda df: df.drop(REGION_NAME),
        lambda v: v.drop(REGION_NAME),
    ),
    # --------------------------------------------------------------- pipelines
    Case(
        "pipeline_filter_then_transform",
        lambda df: df.filter(pl.col(AMOUNT) > 20).with_columns(
            (pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")
        ),
        lambda v: v.filter(pl.col(AMOUNT) > 20).with_columns(
            (pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")
        ),
    ),
    Case(
        "pipeline_transform_then_filter",
        lambda df: df.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")).filter(
            pl.col("region.store.sale.dbl") > 60
        ),
        lambda v: v.with_columns((pl.col(AMOUNT) * 2).alias("region.store.sale.dbl")).filter(
            pl.col("region.store.sale.dbl") > 60
        ),
    ),
    Case(
        "pipeline_cross_level_then_filter",
        lambda df: df.with_columns(
            (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
        ).filter(pl.col("region.store.sale.net") > 40),
        lambda v: v.with_columns(
            (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
        ).filter(pl.col("region.store.sale.net") > 40),
    ),
    Case(
        "pipeline_drop_then_filter",
        lambda df: df.drop(QTY).filter(pl.col(AMOUNT) > 30),
        lambda v: v.drop(QTY).filter(pl.col(AMOUNT) > 30),
    ),
]


# =============================================================================
# The equivalence laws
# =============================================================================


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
class TestNestedEquivalence:
    """view.collect_nested() == pack(flat_with_operation)."""

    def test_nested_matches_packed(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        want = packer.pack(case.flat_op(flat), ROOT)
        got = case.view_op(view).collect_nested()
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
        got = case.view_op(view).collect("sale")
        assert sorted(got.columns) == sorted(want.columns), case.name
        assert_frame_equal(
            got.select(sorted(got.columns)).sort(SALE_ID),
            want.select(sorted(want.columns)).sort(SALE_ID),
        )

    def test_schema_is_stable(
        self, case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """The advertised schema matches what materializing actually produces."""
        result = case.view_op(view)
        assert result.schema == result.collect_nested().schema

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


EVAL_CASES = [c for c in CASES if c.eval_fields is not None]


@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.name)
def test_view_matches_nested_eval(
    case: Case, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
):
    """
    Head-to-head against the nested expression builder.

    ``apply_nested_operations`` transforms fields in place through the
    ``List[Struct]`` layers; the view does the same work as a flat column
    operation. Both must land on identical nested frames.
    """
    packed = packer.pack(flat, ROOT)
    want = apply_nested_operations(
        packed, case.eval_fields, struct_mode=case.eval_mode, use_with_columns=True
    )
    got = case.view_op(view).collect_nested()
    assert_nested_equal(got, want, case.name)


# =============================================================================
# Operations with no single flat equivalent
# =============================================================================


class TestPromoteEquivalence:
    """view.promote() vs packer.promote_attribute()."""

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "first", "last", "list"])
    def test_matches_promote_attribute(
        self, agg, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        want = packer.promote_attribute(
            flat, "amount", from_level="sale", to_level="store", agg=agg, alias="agg"
        ).select(STORE_ID, "region.store.agg")
        got = (
            view.promote("amount", from_level="sale", to_level="store", agg=agg, alias="agg")
            .collect("store")
            .select(STORE_ID, "region.store.agg")
        )
        assert_frame_equal(got.sort(STORE_ID), want.sort(STORE_ID))

    def test_promoted_column_survives_nesting(
        self, view: HierarchyView, packer: HierarchicalPacker, flat: pl.DataFrame
    ):
        promoted = view.promote(
            "amount", from_level="sale", to_level="store", agg="sum", alias="revenue"
        )
        nested = promoted.collect_nested()
        store_struct = nested.schema["region.store"].inner  # type: ignore[union-attr]
        assert "revenue" in store_struct.to_schema()

    def test_promote_to_root(
        self, view: HierarchyView, packer: HierarchicalPacker, flat: pl.DataFrame
    ):
        want = packer.promote_attribute(
            flat, "discount", from_level="store", to_level="region", agg="sum", alias="d"
        ).select(REGION_ID, "region.d")
        got = (
            view.promote("discount", from_level="store", to_level="region", agg="sum", alias="d")
            .collect("region")
            .select(REGION_ID, "region.d")
        )
        assert_frame_equal(got.sort(REGION_ID), want.sort(REGION_ID))


class TestExistentialEquivalence:
    """
    view.any_child_satisfies() vs packer.any_child_satisfies().

    The two APIs spell the predicate differently — the packer evaluates it
    per child struct element (``pl.element().struct.field("amount")``) against
    a frame packed at the parent's granularity, while the view takes the same
    predicate as an ordinary flat column expression. They must select the same
    entities.
    """

    # (flat form for the view, element form for the packer, id)
    CONDITIONS = [
        (pl.col(AMOUNT) > 50, pl.element().struct.field("amount") > 50, "some"),
        (pl.col(AMOUNT) > 1000, pl.element().struct.field("amount") > 1000, "none"),
        (pl.col(AMOUNT) > -1, pl.element().struct.field("amount") > -1, "most"),
        (pl.col(QTY) == 5, pl.element().struct.field("qty") == 5, "exact"),
    ]

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
        # Store granularity: store columns flat, sale still nested — which is
        # exactly what packing to "store" now means.
        at_store = packer.pack(flat, "store")
        want = packer.any_child_satisfies(
            at_store, from_level="sale", to_level="store", condition=element_condition
        )
        got = view.any_child_satisfies(flat_condition, at_level="store", child_level="sale")
        assert sorted(got.tables()["store"].collect()[STORE_ID].to_list()) == sorted(
            want[STORE_ID].to_list()
        )

    @pytest.mark.parametrize(
        "flat_condition", [f for f, _, _ in CONDITIONS], ids=[i for _, _, i in CONDITIONS]
    )
    def test_nested_result_matches_flat_ground_truth(
        self,
        flat_condition,
        flat: pl.DataFrame,
        view: HierarchyView,
        packer: HierarchicalPacker,
    ):
        """Restricting to qualifying stores is the same as filtering flat by them."""
        qualifying = set(flat.filter(flat_condition)[STORE_ID].to_list())
        want = packer.pack(
            flat.filter(pl.col(STORE_ID).is_in(list(qualifying)) if qualifying else pl.lit(False)),
            ROOT,
        )
        got = view.any_child_satisfies(
            flat_condition, at_level="store", child_level="sale"
        ).collect_nested()
        assert_nested_equal(got, want)

    def test_skip_level_has_no_packer_equivalent(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        """The view can skip levels; the packer rejects it. Verify both halves."""
        condition = pl.col(AMOUNT) > 50
        with pytest.raises(ValueError, match="immediate child"):
            packer.any_child_satisfies(
                packer.pack(flat, ROOT), from_level="sale", to_level="region", condition=condition
            )
        kept = view.any_child_satisfies(condition, at_level="region", child_level="sale")
        expected = flat.filter(condition)[REGION_ID].n_unique()
        assert kept.tables()["region"].collect().height == expected


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
            view.filter(self.PREDICATE).collect_nested(),
            packer.pack(flat.filter(self.PREDICATE), ROOT),
        )

    def test_keep_retains_more_children(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        kept = (
            HierarchyView.from_frame(flat, packer, empty_parents="keep")
            .filter(pl.col(REGION_NAME) == "east")
            .collect_nested()
        )
        pruned = (
            HierarchyView.from_frame(flat, packer)
            .filter(pl.col(REGION_NAME) == "east")
            .collect_nested()
        )
        # Same regions either way; 'keep' preserves stores that lost every sale.
        assert kept.height == pruned.height
        assert kept.schema == pruned.schema

    def test_filter_removing_everything(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker
    ):
        impossible = pl.col(AMOUNT) > 10_000
        got = view.filter(impossible).collect_nested()
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
        got = view.collect_nested()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("country.code"), want.sort("country.code"))

    def test_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("country.city.pop") > 2.5
        view = HierarchyView.from_frame(flat, packer)
        want = packer.pack(flat.filter(predicate), "country")
        got = view.filter(predicate).collect_nested()
        assert_frame_equal(got.sort("country.code"), want.sort("country.code"))

    def test_cross_level_transform(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        expr = (pl.col("country.city.pop") * 2).alias("country.city.doubled")
        view = HierarchyView.from_frame(flat, packer)
        want = packer.pack(flat.with_columns(expr), "country")
        got = view.with_columns(expr).collect_nested()
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
        got = HierarchyView.from_frame(flat, packer).collect_nested()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_leaf_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("a.b.c.d.value") > 20
        want = packer.pack(flat.filter(predicate), "a")
        got = HierarchyView.from_frame(flat, packer).filter(predicate).collect_nested()
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_three_hop_ancestor_reference(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        """Leaf 'd' reading root 'a' — three levels up."""
        expr = (pl.col("a.b.c.d.value") * pl.col("a.rate")).alias("a.b.c.d.scaled")
        want = packer.pack(flat.with_columns(expr), "a")
        got = HierarchyView.from_frame(flat, packer).with_columns(expr).collect_nested()
        assert got.schema == want.schema
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))

    def test_three_hop_ancestor_filter(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        predicate = pl.col("a.b.c.d.value") > pl.col("a.rate") * 100
        want = packer.pack(flat.filter(predicate), "a")
        got = HierarchyView.from_frame(flat, packer).filter(predicate).collect_nested()
        assert_frame_equal(got.sort("a.id"), want.sort("a.id"))


# =============================================================================
# Round-trip laws
# =============================================================================


class TestRoundTripLaws:
    """Algebraic identities that must hold regardless of the operation."""

    def test_view_from_packed_frame(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        """A view built from an already-packed frame round-trips to itself."""
        packed = packer.pack(flat, ROOT)
        got = HierarchyView.from_frame(packed, packer).collect_nested()
        assert got.schema == packed.schema
        assert_frame_equal(got.sort(REGION_ID), packed.sort(REGION_ID))

    def test_parquet_round_trip_is_lossless(
        self, flat: pl.DataFrame, view: HierarchyView, packer: HierarchicalPacker, tmp_path
    ):
        view.sink_parquet(tmp_path)
        got = HierarchyView.scan_parquet(tmp_path, packer).collect_nested()
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
        got = view.collect(level)
        assert sorted(got.columns) == sorted(want.columns)
        assert_frame_equal(
            got.select(sorted(got.columns)).sort(key),
            want.select(sorted(want.columns)).sort(key),
        )

    def test_lazy_and_eager_input_agree(self, flat: pl.DataFrame, packer: HierarchicalPacker):
        eager = HierarchyView.from_frame(flat, packer).collect_nested()
        lazy = HierarchyView.from_frame(flat.lazy(), packer).collect_nested()
        assert_frame_equal(eager.sort(REGION_ID), lazy.sort(REGION_ID))

    def test_dangling_child_is_not_silently_dropped_by_augmentation(
        self, flat: pl.DataFrame, packer: HierarchicalPacker
    ):
        """
        Pulling an ancestor column down must not drop rows.

        A cross-level expression joins the ancestor's column onto the child.
        That join has to be a LEFT join: if referential integrity is broken —
        a sale pointing at a store that isn't there — the sale keeps its row
        with a null ancestor value rather than vanishing because of an
        unrelated expression.
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

        before = view.tables()["sale"].collect().height
        after = (
            view.with_columns(
                (pl.col(AMOUNT) * (1 - pl.col(DISCOUNT))).alias("region.store.sale.net")
            )
            .tables()["sale"]
            .collect()
        )
        assert after.height == before, "cross-level expression dropped rows"
        assert after.filter(pl.col(SALE_ID) == 999)["region.store.sale.net"].to_list() == [None]

    def test_tables_and_to_nested_agree(self, view: HierarchyView, packer: HierarchicalPacker):
        """collect_nested() is exactly denormalize(tables())."""
        filtered = view.filter(pl.col(AMOUNT) > 30)
        via_tables = packer.denormalize(filtered.tables()).collect()  # type: ignore[union-attr]
        assert_frame_equal(via_tables.sort(REGION_ID), filtered.collect_nested().sort(REGION_ID))
