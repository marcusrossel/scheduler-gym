import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import plotnine as p9
import polars as pl

from gym import SCHEDULERS, Scheduler

ROOT = Path(__file__).resolve().parent
BENCHDATA = ROOT / "benchdata"
FIGURES = ROOT / "figures"

KEY = ("case", "scheduler", "run")
Resource = Literal["time", "size"]
Metric = Literal["cost", "time", "size"]
# schedulers, or `with_best_k`'s labels
Shown = Scheduler | Literal["lhs-best", "rhs-best"]

RESOURCES: dict[Resource, str] = {"time": "time [s]", "size": "e-graph size"}
METRICS: dict[Metric, str] = {"cost": "cost", **RESOURCES}
# the labels as ordered categories, so facets come in this order
RESOURCE = pl.Enum(RESOURCES.values())
METRIC = pl.Enum(METRICS.values())

ENTRY_RE = re.compile(
    r"costs=\[([^\]]*)\], total_size=(\S+), time=(\S+), iteration=(\d+), stop=(.*)$"
)


def setup_style():
    mpl.rcParams["pdf.fonttype"] = 42
    p9.theme_set(
        p9.theme_minimal()
        + p9.theme(
            panel_grid_minor=p9.element_line(color="#EBEBEB", size=0.5),
            #     panel_grid_minor=p9.element_blank(),
            #     strip_text=p9.element_text(size=9, weight="bold", ha="left"),
            #     legend_title=p9.element_blank(),
            #     plot_title=p9.element_text(size=11, weight="bold", ha="left"),
            #     plot_title_position="plot",
            dpi=300,
        )
    )


def save(plot: p9.ggplot, name: str, size: tuple[float, float]) -> p9.ggplot:
    """Write `plot` to figures/`name`.pdf and return it for display."""
    plot += p9.theme(figure_size=size)
    FIGURES.mkdir(exist_ok=True)
    plot.save(FIGURES / f"{name}.pdf", verbose=False)
    return plot


def load(root: Path = BENCHDATA) -> pl.DataFrame:
    """One row per entry of every scheduler; `run` numbers the runs within a file. Unfinished trailing runs are dropped."""
    rows = []
    for path in sorted(root.glob("*/*.entries")):
        case, scheduler = path.stem, path.parent.name.removesuffix(".rs")
        run, pending = 0, []
        for line in path.read_text().splitlines():
            m = ENTRY_RE.search(line)
            assert m
            costs, size, time, iteration, stop = m.groups()
            cost = sum(float(c) for c in costs.split(", "))
            pending.append(
                (case, scheduler, run, int(iteration), cost, float(size), float(time))
            )
            if stop.strip() != "None":
                rows += pending
                run, pending = run + 1, []
    schema = [*KEY, "iteration", "cost", "size", "time"]
    df = pl.DataFrame(rows, schema=schema, orient="row")
    return df.with_columns(pl.col("scheduler").cast(pl.Enum(SCHEDULERS)))


def select(
    df: pl.DataFrame, schedulers: Sequence[Shown] = SCHEDULERS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows and `finals` of `schedulers`, ordered as given (fixes order and colors in figures)."""
    df = df.filter(pl.col("scheduler").is_in(schedulers)).with_columns(
        pl.col("scheduler").cast(pl.String).cast(pl.Enum(schedulers))
    )
    return df, finals(df)


def finals(df: pl.DataFrame) -> pl.DataFrame:
    """One row per run: final cost, best final cost over all schedulers, and resources needed to reach it."""
    last = df.group_by(KEY, maintain_order=True).last()
    last = last.with_columns(best=pl.col("cost").min().over("case", "run"))

    reached = (
        df.join(last.select(*KEY, "best"), on=KEY)
        .filter(pl.col("cost") <= pl.col("best"))
        .group_by(KEY, maintain_order=True)
        .first()
        .select(*KEY, *(pl.col(r).alias(f"{r}_to_best") for r in RESOURCES))
    )
    return last.join(reached, on=KEY, how="left")


def check_runs(df: pl.DataFrame, schedulers: Sequence[Shown] = SCHEDULERS):
    """Print the cases where `schedulers` have a different number of finished runs."""
    _, fin = select(df, schedulers)
    counts = fin.group_by("case", "scheduler").len()
    for (case,), g in counts.group_by("case"):
        if g["len"].n_unique() > 1:
            print(f"{case}: schedulers disagree on number of runs:")
            print(g.sort("len").select("scheduler", "len"))


def _geomean(e: pl.Expr) -> pl.Expr:
    return e.log().mean().exp()


def _titled(d: pl.DataFrame, fin: pl.DataFrame) -> pl.DataFrame:
    """`d` with `title` = case and its number of runs, for facet labels."""
    runs = fin.group_by("case").agg(
        title=pl.format(
            "{} ({} runs)", pl.col("case").first(), pl.col("run").n_unique()
        )
    )
    return d.join(runs, on="case")


def _facets(by: str, n: int, scales: Literal["free", "free_x"]) -> p9.facet_wrap:
    """One column per case (`title`), one row per value of `by`, of which there are `n`."""
    return p9.facet_wrap(
        ["title", by],
        nrow=n,
        dir="v",
        scales=scales,
        # plotnine annotates `labeller` too narrowly, facet accepts any labeller
        labeller=p9.labeller(multi_line=False),  # ty: ignore[invalid-argument-type]
    )


def _stars(d: pl.DataFrame, by: list[str], y: str) -> pl.DataFrame:
    """Per group of `by` and scheduler, the first point where `y` reaches its final value."""
    group = [*by, "scheduler"]
    last = pl.col(y).last().over(group)
    return (
        d.sort("x", maintain_order=True)
        .filter((pl.col(y) - last).abs() <= 1e-9 * last.abs())
        .group_by(group)
        .first()
    )


def _steps(
    d: pl.DataFrame, y: str, by: list[str], log_x: bool, star_if: pl.Expr | bool = True
) -> p9.ggplot:
    """Step lines of `y` over `x` per group of `by` and scheduler, with a star where
    each curve reaches its final value (only for the stars matching `star_if`)."""
    stars = _stars(d, by, y).filter(star_if)
    return (
        p9.ggplot(d, p9.aes("x", y, color="scheduler"))
        + p9.geom_step(size=0.8)
        # filled in the curve's color, with a black edge to stand out from the curve
        + p9.geom_point(
            p9.aes(fill="scheduler"),
            data=stars,
            shape="*",
            size=6,
            color="black",
            stroke=0.3,
            show_legend=False,
        )
        + (p9.scale_x_log10() if log_x else p9.scale_x_continuous())
        + p9.scale_color_hue()
        + p9.scale_fill_hue()
    )


def _bars(d: pl.DataFrame, value: str, fmt: str, *, base: float = 0) -> p9.ggplot:
    """Horizontal bars per scheduler from `base` to `value`, each labelled with its value."""
    dtype = d["scheduler"].dtype
    assert isinstance(dtype, pl.Enum)
    top_down = dtype.categories.reverse().to_list()
    d = d.with_columns(base=pl.lit(base))
    return (
        p9.ggplot(d, p9.aes(y="scheduler", x="base", xend=value, color="scheduler"))
        + p9.geom_segment(p9.aes(yend="scheduler"), size=4)
        + p9.geom_text(
            p9.aes(x=value, label=value),
            format_string=f" {{:{fmt}}}",
            ha="left",
            size=7,
        )
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:.3g}" for x in xs], expand=(0, 0, 0.25, 0)
        )
        + p9.scale_y_discrete(limits=top_down)
        + p9.scale_color_hue()
        + p9.guides(color="none")
        + p9.labs(y="")
        + p9.theme(panel_grid_major_y=p9.element_blank())
    )


def ranking(df: pl.DataFrame, schedulers: Sequence[Shown] = SCHEDULERS) -> p9.ggplot:
    """Geometric mean of final cost relative to the best scheduler per run (1 = always best)."""
    _, fin = select(df, schedulers)
    d = fin.group_by("case", "scheduler").agg(
        ratio=_geomean(pl.col("cost") / pl.col("best"))
    )
    return (
        _bars(_titled(d, fin), "ratio", ".3f", base=1)
        + p9.facet_wrap("title", nrow=1, scales="free")
        + p9.labs(
            x="final cost / best final cost (geomean)",
            title="Final cost relative to the best scheduler",
            subtitle="geometric mean over runs, 1 = always the best; lower is better",
        )
    )


def sums(df: pl.DataFrame, schedulers: Sequence[Shown] = SCHEDULERS) -> p9.ggplot:
    """Sum of the final cost, time and size over all runs, like dump.py's dumpall."""
    _, fin = select(df, schedulers)
    d = pl.concat(
        fin.group_by("case", "scheduler").agg(
            metric=pl.lit(label, METRIC), total=pl.col(m).sum()
        )
        for m, label in METRICS.items()
    )
    return (
        _bars(_titled(d, fin), "total", ".4g")
        + _facets("metric", len(METRICS), "free")
        + p9.labs(
            x="total final value",
            title="Totals over all runs",
            subtitle="sum of the final cost, time and e-graph size; lower is better",
        )
    )


def solved(
    df: pl.DataFrame,
    schedulers: Sequence[Shown] = SCHEDULERS,
    *,
    case: str = "caviar",
    solved_cost: float = 1,
) -> p9.ggplot:
    """Number of runs per scheduler that ended at `solved_cost`, i.e. proved the goal."""
    _, fin = select(df, schedulers)
    d = (
        fin.filter(pl.col("case") == case)
        .group_by("case", "scheduler")
        .agg(solved=(pl.col("cost") == solved_cost).sum(), runs=pl.len())
    )
    runs = d["runs"].max()
    return (
        _bars(d, "solved", "d")
        + p9.expand_limits(x=runs)
        + p9.labs(
            x=f"solved runs (of {runs})",
            title=f"{case}: proved goals",
            subtitle="runs that ended at the goal; higher is better",
        )
    )


def with_best_k(
    df: pl.DataFrame,
    ks: dict[str, dict[str, int]],
    base: Sequence[Scheduler] = ("backoff",),
) -> pl.DataFrame:
    """`base` plus each case's detour-<side>-<k> from `ks` ({case: {side: k}}) as <side>-best."""
    picks = {
        f"{case}/detour-{side}-{k}": f"{side}-best"
        for case, sides in ks.items()
        for side, k in sides.items()
    }
    df = df.filter(pl.col("case").is_in(ks.keys())).with_columns(
        pl.col("scheduler").cast(pl.String)
    )
    label = pl.concat_str("case", pl.lit("/"), "scheduler").replace_strict(
        picks, default=None
    )
    return df.filter(
        label.is_not_null() | pl.col("scheduler").is_in(base)
    ).with_columns(
        scheduler=pl.coalesce(label, "scheduler"),
        case=pl.col("case").replace(
            {
                case: f"{case}: "
                + ", ".join(f"{side} {k}" for side, k in sides.items())
                for case, sides in ks.items()
            }
        ),
    )


def suggest_best_k(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    solved_case: str = "caviar",
    solved_cost: float = 1,
) -> pl.DataFrame:
    """The lhs/rhs ks per case side by side, ranked as in `ranking` (by solved runs first for
    `solved_case`, the only case with a solved count)."""
    _, fin = select(df, schedulers)
    side_k = (
        pl.col("scheduler")
        .cast(pl.String)
        .str.extract_groups(r"^detour-(?P<side>lhs|rhs)-(?P<k>\d+)$")
    )
    d = (
        fin.with_columns(
            side=side_k.struct["side"], k=side_k.struct["k"].cast(pl.Int64)
        )
        .filter(pl.col("side").is_not_null())
        .group_by("case", "side", "k")
        .agg(
            ratio=_geomean(pl.col("cost") / pl.col("best")),
            solved=(
                (pl.col("case") == solved_case) & (pl.col("cost") == solved_cost)
            ).sum(),
        )
        # ties go to the smallest k
        .sort(
            "case",
            "side",
            "solved",
            "ratio",
            "k",
            descending=[False, False, True, False, False],
        )
        .with_columns(
            rank=pl.int_range(1, pl.len() + 1).over("case", "side"),
            ratio=pl.col("ratio").round(3),
            solved=pl.when(pl.col("case") == solved_case).then("solved"),
        )
    )
    sides = d["side"].unique(maintain_order=True)
    return (
        d.pivot(on="side", index=["case", "rank"], values=["k", "ratio", "solved"])
        .select(
            "case",
            "rank",
            *(f"{c}_{side}" for side in sides for c in ("k", "ratio", "solved")),
        )
        .sort("case", "rank", maintain_order=True)
    )


def _per_resource(df: pl.DataFrame, *cols: str) -> pl.DataFrame:
    """`df` once per resource, with `x` its running maximum (the e-graph size can shrink)."""
    return pl.concat(
        df.select(
            *KEY,
            *cols,
            resource=pl.lit(label, RESOURCE),
            x=pl.col(r).cum_max().over(KEY),
        )
        for r, label in RESOURCES.items()
    )


def _profile(
    points: pl.DataFrame,
    ends: pl.DataFrame,
    fin: pl.DataFrame,
    log_x: bool,
    y_min: float,
    nrow: int = len(RESOURCES),
) -> p9.ggplot:
    """Step curves from 0 at `lo` through `points` to `hi`, per case, resource and scheduler,
    showing y from `y_min` to 1, in `nrow` rows of facets."""
    group = ["case", "resource", "scheduler"]
    curves = pl.concat(
        [
            ends.select(*group, x="lo", y=pl.lit(0.0)),
            points.select(*group, "x", "y"),
            ends.select(*group, x="hi", y="y"),
        ]
    ).sort("x", maintain_order=True)
    curves = _titled(curves, fin)
    return (
        # curves that stay at 0 get no star
        _steps(curves, "y", ["case", "resource"], log_x, pl.col("y") > 0)
        + _facets("resource", nrow, "free_x")
        + p9.coord_cartesian(ylim=(y_min, 1))
    )


def solved_profile(
    df: pl.DataFrame,
    schedulers: Sequence[Shown] = SCHEDULERS,
    *,
    case: str = "caviar",
    solved_cost: float = 1,
    log_x: bool = False,
    y_min: float = 0,
) -> p9.ggplot:
    """Fraction of runs per scheduler that reached `solved_cost` within a resource budget.
    `log_x`: log x axis; `y_min`: lower end of the y axis, to zoom in on high fractions."""
    df, fin = select(df.filter(pl.col("case") == case), schedulers)
    group = ["case", "resource", "scheduler"]
    d = _per_resource(df, "cost")
    runs = fin.group_by("case", "scheduler").agg(runs=pl.len())
    proved = pl.col("cost") <= solved_cost
    proofs = (
        d.filter(proved)
        .group_by(*KEY, "resource", maintain_order=True)
        .agg(pl.col("x").first())
        .join(runs, on=["case", "scheduler"])
        .with_columns(y=pl.col("x").rank("ordinal").over(group) / pl.col("runs"))
    )
    # without any proof, the curves start at the smallest resource used at all
    lo = d.group_by("case", "resource").agg(
        lo=pl.col("x").filter(proved).min().fill_null(pl.col("x").min())
    )
    ends = (
        d.group_by(group)
        .agg(hi=pl.col("x").max())
        .join(lo, on=["case", "resource"])
        .join(proofs.group_by(group).agg(pl.col("y").max()), on=group, how="left")
        .with_columns(pl.col("y").fill_null(0))
    )
    # a single case, so the resources go side by side
    return _profile(proofs, ends, fin, log_x, y_min, nrow=1) + p9.labs(
        x="",
        y="runs with a proof",
        title=f"{case}: proving the goal",
        subtitle="fraction of runs that proved the goal within a time / e-graph size"
        " budget;\na curve ends at the most its scheduler used, ★ marks its last proof;"
        " higher is better",
    )


def profile(
    df: pl.DataFrame,
    schedulers: Sequence[Shown] = SCHEDULERS,
    *,
    log_x: bool = False,
    y_min: float = 0,
) -> p9.ggplot:
    """Fraction of runs per scheduler that reached the best known cost within a resource budget.
    `log_x`: log x axis; `y_min`: lower end of the y axis, to zoom in on high fractions."""
    _, fin = select(df, schedulers)
    group = ["case", "resource", "scheduler"]
    d = pl.concat(
        fin.select(
            "case",
            "scheduler",
            resource=pl.lit(label, RESOURCE),
            x=pl.col(f"{res}_to_best"),
            hi=pl.col(res).max().over("case"),
        )
        for res, label in RESOURCES.items()
    ).with_columns(
        lo=pl.col("x").min().over("case", "resource"),
        # runs that never reached the best have a null `x`, and so a null rank
        y=(pl.col("x").rank("ordinal") / pl.len()).over(group),
    )
    ends = d.group_by(group).agg(
        pl.col("lo", "hi").first(), pl.col("y").max().fill_null(0)
    )
    return _profile(d.drop_nulls("x"), ends, fin, log_x, y_min) + p9.labs(
        x="",
        y="runs at best known cost",
        title="Reaching the best known cost",
        subtitle="fraction of runs that reached the best final cost of any scheduler"
        " within a time / e-graph size budget;\n★ marks where a curve stops rising;"
        " higher is better",
    )


def _relative(
    df: pl.DataFrame, fin: pl.DataFrame, metric: Metric, absolute: bool = False
) -> pl.DataFrame:
    """`df` with `rel` = `metric` / the run's smallest final `metric`, or `metric` if `absolute`."""
    if absolute:
        return df.with_columns(rel=pl.col(metric))
    best = fin.group_by("case", "run").agg(best=pl.col(metric).min())
    return df.join(best, on=["case", "run"]).with_columns(
        rel=pl.col(metric) / pl.col("best")
    )


def _grid(d: pl.DataFrame, by: list[str], log: bool, n: int = 300) -> pl.DataFrame:
    """`n` evenly spaced `x` per group of `by`, spanning that group's `x`; log-spaced if `log`."""
    lo, hi = pl.col("x").min(), pl.col("x").max()
    if log:
        lo, hi = lo.log(), hi.log()
    grid = d.group_by(by).agg(x=pl.linear_spaces(lo, hi, n)).explode("x")
    return grid.with_columns(pl.col("x").exp()) if log else grid


def _geomean_curves(d: pl.DataFrame, by: list[str], grid: pl.DataFrame) -> pl.DataFrame:
    """Per group of `by` and scheduler, the geomean over runs of `rel` at each `x` of `grid`.
    Outside its range, a run is held at its first / last `rel`."""
    run = [*by, "scheduler", "run"]
    d = d.select(*run, "x", "rel").sort("x", maintain_order=True)
    starts = d.group_by(run).agg(start=pl.col("rel").first())
    return (
        grid.join(starts, on=by)
        .sort("x")
        .join_asof(d, on="x", by=run, check_sortedness=False)
        .group_by(*by, "scheduler", "x")
        .agg(rel=_geomean(pl.col("rel").fill_null(pl.col("start"))))
    )


def quality(
    df: pl.DataFrame,
    schedulers: Sequence[Shown] = SCHEDULERS,
    *,
    log_x: bool = False,
    log_y: bool = False,
    absolute: bool = False,
) -> p9.ggplot:
    """Geomean over runs of the cost relative to the run's best final cost, over resource.
    `log_x`/`log_y`: log axes; `absolute`: plain cost (only the y scale changes)."""
    df, fin = select(df, schedulers)
    d = _per_resource(_relative(df, fin, "cost", absolute), "rel")
    by = ["case", "resource"]
    curves = _titled(_geomean_curves(d, by, _grid(d, by, log_x)), fin)
    return (
        _steps(curves, "rel", by, log_x)
        + _facets("resource", len(RESOURCES), "free")
        + (p9.scale_y_log10() if log_y else p9.scale_y_continuous())
        + p9.labs(
            x="",
            y="cost (geomean)" if absolute else "cost / best final cost (geomean)",
            title="Cost over time and e-graph size",
            subtitle=(
                "cost"
                if absolute
                else "cost relative to the best final cost of each run"
            )
            + ", geometric mean over runs;\n★ marks where a curve stops improving;"
            " lower is better",
        )
    )


def dev(
    df: pl.DataFrame,
    schedulers: Sequence[Shown] = SCHEDULERS,
    *,
    log_x: bool = False,
    log_y: bool = False,
    absolute: bool = False,
) -> p9.ggplot:
    """Geomean over runs of cost, time and size relative to the run's smallest final value.
    `log_x`/`log_y`: log axes; `absolute`: plain values (only the y scale changes)."""
    df, fin = select(df, schedulers)
    d = pl.concat(
        _relative(df, fin, m, absolute).select(
            *KEY,
            "rel",
            metric=pl.lit(label, METRIC),
            x=pl.col("iteration").cast(pl.Float64),
        )
        for m, label in METRICS.items()
    )
    by = ["case", "metric"]
    grid = _grid(d, by, log_x).with_columns(pl.col("x").round()).unique()
    curves = _titled(_geomean_curves(d, by, grid), fin)
    return (
        # time and size only grow, so only the cost gets stars
        _steps(curves, "rel", by, log_x, pl.col("metric") == METRICS["cost"])
        + _facets("metric", len(METRICS), "free")
        + (p9.scale_y_log10() if log_y else p9.scale_y_continuous())
        + p9.labs(
            x="iteration",
            y="value (geomean)"
            if absolute
            else "value / smallest final value (geomean)",
            title="Development per iteration",
            subtitle="cost, time and e-graph size"
            + ("" if absolute else " relative to the smallest final value of each run")
            + ", geometric mean over runs;\n★ marks where the cost stops improving;"
            " lower is better",
        )
    )
