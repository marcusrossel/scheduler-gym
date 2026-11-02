import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import plotnine as p9
import polars as pl

from gym import SCHEDULERS, Scheduler

ROOT = Path(__file__).resolve().parent
BENCHDATA = ROOT / "benchdata-old"
FIGURES = ROOT / "figures"

KEY = ("case", "scheduler", "run")
Resource = Literal["time", "size"]
Metric = Literal["cost", "time", "size"]
Side = Literal["backoff", "lhs", "rhs", "other"]
COLORS: dict[Side, str] = {
    "backoff": "black",
    "lhs": "#E69F00",
    "rhs": "#0072B2",
    "other": "#999999",
}
SIDE = pl.Enum(COLORS.keys())
# alpha of the curves and bars that are not the best of their side
REST_ALPHA = 0.4
TOP_NOTE = "opaque: backoff and the best lhs / rhs of each case"

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
    df: pl.DataFrame, schedulers: Sequence[Scheduler] = SCHEDULERS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows and `finals` of `schedulers`, ordered as given (fixes order in figures),
    the finals with each scheduler's `side` and whether it is the `top` of it per case."""
    df = df.filter(pl.col("scheduler").is_in(schedulers)).with_columns(
        pl.col("scheduler").cast(pl.String).cast(pl.Enum(schedulers))
    )
    fin = finals(df)
    name = pl.col("scheduler").cast(pl.String)
    side = name.str.extract(r"^detour-(lhs|rhs)-\d+$")
    # backoff is always top; of lhs / rhs the best as in `ranking`, ties going to the
    # smallest k
    top = (
        fin.group_by("case", "scheduler")
        .agg(ratio=_geomean(pl.col("cost") / pl.col("best")))
        .with_columns(
            side=pl.when(name == "backoff")
            .then(pl.lit("backoff"))
            .otherwise(side.fill_null("other"))
            .cast(SIDE),
            k=name.str.extract(r"-(\d+)$").cast(pl.Int64),
        )
        .sort("ratio", "k")
        .select(
            "case",
            "scheduler",
            "side",
            top=pl.when(pl.col("side").is_in(["lhs", "rhs"]))
            .then(pl.int_range(pl.len()).over("case", "side") == 0)
            .otherwise(pl.col("side") == "backoff"),
        )
    )
    return df, fin.join(top, on=["case", "scheduler"], how="left")


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


def check_runs(df: pl.DataFrame, schedulers: Sequence[Scheduler] = SCHEDULERS):
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
    """`d` with `title` = case, number of runs and top lhs / rhs, for facet labels."""
    top = pl.col("top") & pl.col("side").is_in(["lhs", "rhs"])
    pick = pl.col("scheduler").cast(pl.String).str.replace(r"^detour-(\w+)-", "$1 ")
    runs = fin.group_by("case").agg(
        title=pl.format(
            "{} ({} runs) · {}",
            pl.col("case").first(),
            pl.col("run").n_unique(),
            pick.filter(top).unique().sort().str.join(", "),
        ).str.strip_suffix(" · ")
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
    d: pl.DataFrame,
    fin: pl.DataFrame,
    y: str,
    by: list[str],
    log_x: bool,
    star_if: pl.Expr | bool = True,
) -> p9.ggplot:
    """Step lines of `y` over `x` per group of `by` and scheduler, colored by side, the top
    ones opaque with a star where they reach their final value (if matching `star_if`)."""
    d = d.join(
        fin.select("case", "scheduler", "side", "top").unique(),
        on=["case", "scheduler"],
    )
    top = d.filter(pl.col("top"))
    # legend with only the sides present
    breaks = [side for side in COLORS if side in d["side"]]
    stars = _stars(d, by, y).filter(star_if & pl.col("top"))
    return (
        p9.ggplot(d, p9.aes("x", y, color="side", group="scheduler"))
        + p9.geom_step(data=d.filter(~pl.col("top")), size=0.5, alpha=REST_ALPHA)
        # a white halo, so the top curves stand out where they cross the rest
        + p9.geom_step(data=top, size=2.4, color="white")
        + p9.geom_step(data=top, size=1.4)
        # filled in the curve's color, with a black edge to stand out from the curve
        + p9.geom_point(
            p9.aes(fill="side"),
            data=stars,
            shape="*",
            size=6,
            color="black",
            stroke=0.3,
            show_legend=False,
        )
        + (p9.scale_x_log10() if log_x else p9.scale_x_continuous())
        + p9.scale_color_manual(values=COLORS, breaks=breaks, name="scheduler")
        + p9.scale_fill_manual(values=COLORS, breaks=breaks)
    )


def _bars(
    d: pl.DataFrame, fin: pl.DataFrame, value: str, fmt: str, *, base: float = 0
) -> p9.ggplot:
    """Horizontal bars per scheduler from `base` to `value`, labelled with it, colored by
    side, the top ones opaque."""
    dtype = d["scheduler"].dtype
    assert isinstance(dtype, pl.Enum)
    top_down = dtype.categories.reverse().to_list()
    d = d.join(
        fin.select("case", "scheduler", "side", "top").unique(),
        on=["case", "scheduler"],
    ).with_columns(base=pl.lit(base))
    bar = p9.aes(yend="scheduler")
    return (
        p9.ggplot(d, p9.aes(y="scheduler", x="base", xend=value, color="side"))
        + p9.geom_segment(bar, data=d.filter(~pl.col("top")), size=4, alpha=REST_ALPHA)
        + p9.geom_segment(bar, data=d.filter(pl.col("top")), size=4)
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
        + p9.scale_color_manual(values=COLORS)
        + p9.guides(color="none")
        + p9.labs(y="")
        + p9.theme(panel_grid_major_y=p9.element_blank())
    )


def ranking(
    df: pl.DataFrame, schedulers: Sequence[Scheduler] = SCHEDULERS
) -> p9.ggplot:
    """Geometric mean of final cost relative to the best scheduler per run (1 = always best)."""
    _, fin = select(df, schedulers)
    d = fin.group_by("case", "scheduler").agg(
        ratio=_geomean(pl.col("cost") / pl.col("best"))
    )
    return (
        _bars(_titled(d, fin), fin, "ratio", ".3f", base=1)
        + p9.facet_wrap("title", nrow=1, scales="free")
        + p9.labs(
            x="final cost / best final cost (geomean)",
            title="Final cost relative to the best scheduler",
            subtitle="geometric mean over runs, 1 = always the best; lower is better;"
            f"\n{TOP_NOTE}",
        )
    )


def solved(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    solved_cost: float = 1,
) -> p9.ggplot:
    """Number of runs per scheduler that ended at `solved_cost`, i.e. proved the goal, per
    case of `df`, which are all proof cases."""
    _, fin = select(df, schedulers)
    d = fin.group_by("case", "scheduler").agg(
        solved=(pl.col("cost") == solved_cost).sum(), runs=pl.len()
    )
    return (
        _bars(_titled(d, fin), fin, "solved", "d")
        # each case's axis reaches its number of runs
        + p9.geom_blank(p9.aes(x="runs"))
        + p9.facet_wrap("title", nrow=1, scales="free")
        + p9.labs(
            x="solved runs",
            title="Proved goals",
            subtitle=f"runs that ended at the goal; higher is better;\n{TOP_NOTE}",
        )
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
        _steps(curves, fin, "y", ["case", "resource"], log_x, pl.col("y") > 0)
        + _facets("resource", nrow, "free_x")
        + p9.coord_cartesian(ylim=(y_min, 1))
    )


def solved_profile(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    solved_cost: float = 1,
    log_x: bool = False,
    y_min: float = 0,
) -> p9.ggplot:
    """Fraction of runs per scheduler that reached `solved_cost` within a resource budget,
    per case of `df`, which are all proof cases.
    `log_x`: log x axis; `y_min`: lower end of the y axis, to zoom in on high fractions."""
    df, fin = select(df, schedulers)
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
    # few proof cases, so the resources go side by side
    return _profile(proofs, ends, fin, log_x, y_min, nrow=1) + p9.labs(
        x="",
        y="runs with a proof",
        title="Proving the goal",
        subtitle="fraction of runs that proved the goal within a time / e-graph size"
        " budget; higher is better;\na curve ends at the most its scheduler used,"
        f" ★ marks its last proof;\n{TOP_NOTE}",
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
    schedulers: Sequence[Scheduler] = SCHEDULERS,
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
        _steps(curves, fin, "rel", by, log_x)
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
            + ", geometric mean over runs; lower is better;"
            f"\n★ marks where a curve stops improving; {TOP_NOTE}",
        )
    )


def dev(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
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
        _steps(curves, fin, "rel", by, log_x, pl.col("metric") == METRICS["cost"])
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
            + ", geometric mean over runs; lower is better;"
            f"\n★ marks where the cost stops improving; {TOP_NOTE}",
        )
    )
