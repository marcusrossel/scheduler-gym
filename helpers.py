import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import plotnine as p9
import polars as pl
from matplotlib.path import Path as MplPath

from gym import SCHEDULERS, Scheduler

ROOT = Path(__file__).resolve().parent
BENCHDATA = ROOT / "benchdata"
FIGURES = ROOT / "figures"

KEY = ("case", "scheduler", "run")
Resource = Literal["time", "size"]
Metric = Literal["cost", "time", "size"]
Side = Literal["backoff", "lhs", "rhs", "other"]
COLORS: dict[Side, str] = {
    "backoff": "black",
    "lhs": "#E69F00",
    "rhs": "#0072B2",
    "other": "#CC79A7",
}
SIDE = pl.Enum(COLORS.keys())
_NAME = pl.col("scheduler").cast(pl.String)
SIDE_OF = (
    pl.when(_NAME == "backoff")
    .then(pl.lit("backoff"))
    .otherwise(_NAME.str.extract(r"^detour-(lhs|rhs)-\d+$").fill_null("other"))
    .cast(SIDE)
)
K_OF = _NAME.str.extract(r"-(\d+)$").cast(pl.Int64)
DETOUR = pl.col("side").is_in(["lhs", "rhs"])
# the scheduler the bar charts are relative to
BASELINE: Scheduler = "backoff"
# alpha of the curves and bars that are not the best of their side
REST_ALPHA = 0.4
# marker for the break in cut bars: two slanted parallel strokes, like //
BREAK = MplPath(
    [(-0.6, -1), (-0.1, 1), (0.1, -1), (0.6, 1)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.MOVETO, MplPath.LINETO],
)
TOP_NOTE = "opaque: backoff and the lhs / rhs with the lowest final cost of each case"
BAR_NOTE = (
    "opaque: the lhs / rhs with the lowest final cost of each case;\n"
    f"// = bar cut off; the line at 0 is {BASELINE}"
)

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
    """Write `plot` to figures/`name`.pdf and return it for display.

    Also writes figures/`name`-no-title.pdf without title and subtitle, for use
    in a tex file, and prints them so they can go into the caption.
    """
    plot += p9.theme(figure_size=size)
    FIGURES.mkdir(exist_ok=True)
    plot.save(FIGURES / f"{name}.pdf", verbose=False)
    bare = plot + p9.theme(
        plot_title=p9.element_blank(), plot_subtitle=p9.element_blank()
    )
    bare.save(FIGURES / f"{name}-no-title.pdf", verbose=False)
    print(f"{name}\n{plot.labels.get('title', '')}\n{plot.labels.get('subtitle', '')}")
    return plot


def load(*, root: Path = BENCHDATA) -> pl.DataFrame:
    """One row per entry; `run` numbers the runs in a file, unfinished ones dropped."""
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
    df: pl.DataFrame, *, schedulers: Sequence[Scheduler] = SCHEDULERS
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rows of `schedulers` in that order, and finals: one row per run with the final
    cost, the run's best final cost, resources to reach it, `side` and `top`."""
    df = df.filter(pl.col("scheduler").is_in(schedulers)).with_columns(
        pl.col("scheduler").cast(pl.String).cast(pl.Enum(schedulers))
    )
    fin = df.group_by(KEY, maintain_order=True).last()
    fin = fin.with_columns(best=pl.col("cost").min().over("case", "run"))
    reached = (
        df.join(fin.select(*KEY, "best"), on=KEY)
        .filter(pl.col("cost") <= pl.col("best"))
        .group_by(KEY, maintain_order=True)
        .first()
        .select(*KEY, *(pl.col(r).alias(f"{r}_to_best") for r in RESOURCES))
    )
    fin = fin.join(reached, on=KEY, how="left")

    # backoff is always top; of lhs / rhs the one with the lowest geomean of final cost
    # relative to the best final cost of each run, ties going to the smallest k
    top = (
        fin.group_by("case", "scheduler")
        .agg(ratio=_geomean(pl.col("cost") / pl.col("best")))
        .with_columns(side=SIDE_OF)
        .sort("ratio", K_OF)
        .select("case", "scheduler", "side", top=_top("case"))
    )
    return df, fin.join(top, on=["case", "scheduler"], how="left")


def check_runs(df: pl.DataFrame, *, schedulers: Sequence[Scheduler] = SCHEDULERS):
    """Print cases where `schedulers` have different numbers of finished runs."""
    _, fin = select(df, schedulers=schedulers)
    counts = fin.group_by("case", "scheduler").len()
    for (case,), g in counts.group_by("case"):
        if g["len"].n_unique() > 1:
            print(f"{case}: schedulers disagree on number of runs:")
            print(g.sort("len").select("scheduler", "len"))


def _geomean(e: pl.Expr) -> pl.Expr:
    return e.log().mean().exp()


def _top(*over: str) -> pl.Expr:
    """Backoff, and of lhs / rhs the first row per `over` and side."""
    return (
        pl.when(DETOUR)
        .then(pl.int_range(pl.len()).over(*over, "side") == 0)
        .otherwise(pl.col("side") == "backoff")
    )


def _titled(
    d: pl.DataFrame, fin: pl.DataFrame, tops: pl.DataFrame | None = None
) -> pl.DataFrame:
    """`d` with `side`, `top` and a facet `title`: case, runs, top lhs / rhs.
    `tops`: `side` and `top` per scheduler and further keys, instead of `fin`'s."""
    if tops is None:
        tops = fin.select("case", "scheduler", "side", "top").unique()
    by = [c for c in tops.columns if c not in ("scheduler", "side", "top")]
    top = pl.col("top") & DETOUR
    pick = _NAME.str.replace(r"^detour-(\w+)-", "$1 ")
    runs = fin.group_by("case").agg(runs=pl.col("run").n_unique())
    titles = (
        tops.join(runs, on="case")
        .group_by(by)
        .agg(
            title=pl.format(
                "{} ({} runs) · {}",
                pl.col("case").first(),
                pl.col("runs").first(),
                pick.filter(top).unique().sort().str.join(", "),
            ).str.strip_suffix(" · ")
        )
    )
    return d.join(tops, on=[*by, "scheduler"]).join(titles, on=by)


def _facets(by: str, n: int, scales: Literal["free", "free_x"]) -> p9.facet_wrap:
    """One column per case, `n` rows, one per value of `by`."""
    return p9.facet_wrap(
        ["title", by],
        nrow=n,
        dir="v",
        scales=scales,
        # plotnine annotates `labeller` too narrowly, facet accepts any labeller
        labeller=p9.labeller(multi_line=False),  # ty: ignore[invalid-argument-type]
    )


def _steps(
    d: pl.DataFrame,
    y: str,
    by: list[str],
    log_x: bool,
    *,
    star_if: pl.Expr | bool = True,
) -> p9.ggplot:
    """Step lines per scheduler, top ones opaque, starred where final if `star_if`."""
    top = d.filter(pl.col("top"))
    # legend with only the sides present
    breaks = [side for side in COLORS if side in d["side"]]
    # the first point per group and scheduler where `y` reaches its final value
    group = [*by, "scheduler"]
    last = pl.col(y).last().over(group)
    stars = (
        d.sort("x", maintain_order=True)
        .filter((pl.col(y) - last).abs() <= 1e-9 * last.abs())
        .group_by(group)
        .first()
        .filter(star_if & pl.col("top"))
    )
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


def _bars(d: pl.DataFrame, value: str, *, zoom: float = 100) -> p9.ggplot:
    """Bars of `value` per scheduler, cut at ±`zoom` with a // break.
    Backoff is left out, it is the line at 0."""
    dtype = d["scheduler"].dtype
    assert isinstance(dtype, pl.Enum)
    top_down = [s for s in dtype.categories.reverse() if s != BASELINE]
    gain = pl.col(value) >= 0
    # room for the labels past the bars, about a third of the case's range of bars
    span = pl.max_horizontal(pl.col("end").max(), 0) - pl.min_horizontal(
        pl.col("end").min(), 0
    )
    d = (
        d.filter(pl.col("scheduler") != BASELINE)
        .with_columns(end=pl.col(value).clip(-zoom, zoom))
        .with_columns(
            reach=pl.col("end")
            + pl.when(gain).then(0.3).otherwise(-0.3) * span.over("case"),
            # where a cut bar is broken, a bit before its end
            gap=pl.col("end")
            - pl.when(gain).then(0.08).otherwise(-0.08) * span.over("case"),
        )
    )
    d = d.with_columns(
        text=pl.Series(f" {v:+.1f}%" if v >= 0 else f"{v:+.1f}% " for v in d[value])
    )
    bar = p9.aes(yend="scheduler")
    label = p9.aes(x="end", label="text")
    return (
        p9.ggplot(d, p9.aes(y="scheduler", x=0, xend="end", color="side"))
        + p9.geom_vline(xintercept=0, color=COLORS["backoff"], size=0.5)
        + p9.geom_blank(p9.aes(x="reach"))
        + p9.geom_segment(bar, data=d.filter(~pl.col("top")), size=4, alpha=REST_ALPHA)
        + p9.geom_segment(bar, data=d.filter(pl.col("top")), size=4)
        + p9.geom_point(
            p9.aes(x="gap"),
            data=d.filter(pl.col(value).abs() > zoom),
            shape=BREAK,
            size=5,
            stroke=1.2,
            color="white",
        )
        + p9.geom_text(label, data=d.filter(gain), ha="left", size=7, color="black")
        + p9.geom_text(label, data=d.filter(~gain), ha="right", size=7, color="black")
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:+.3g}%" if x else "0%" for x in xs],
            expand=(0.02, 0),
        )
        + p9.scale_y_discrete(limits=top_down)
        + p9.scale_color_manual(values=COLORS)
        + p9.guides(color="none")
        + p9.labs(y="")
        + p9.theme(panel_grid_major_y=p9.element_blank())
    )


def ranking(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    zoom: float = 100,
) -> p9.ggplot:
    """Final cost reduction over backoff in percent, geomean over runs.
    `zoom`: bars are cut off at ±`zoom` percent, their labels still give the value."""
    _, fin = select(df, schedulers=schedulers)
    base = fin.filter(pl.col("scheduler") == BASELINE).select(
        "case", "run", base=pl.col("cost")
    )
    d = (
        fin.join(base, on=["case", "run"])
        .group_by("case", "scheduler")
        .agg(gain=(1 - _geomean(pl.col("cost") / pl.col("base"))) * 100)
    )
    return (
        _bars(_titled(d, fin), "gain", zoom=zoom)
        + p9.facet_wrap("title", nrow=1, scales="free")
        + p9.labs(
            x=f"final cost reduction over {BASELINE} [%]",
            title=f"Final cost relative to {BASELINE}",
            subtitle=f"1 − geomean over runs of final cost / {BASELINE}'s final cost;"
            f" higher is better;\n{BAR_NOTE}",
        )
    )


def solved(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    solved_cost: float = 1,
    zoom: float = 100,
) -> p9.ggplot:
    """Change in runs that reached `solved_cost`, relative to backoff in percent.
    `zoom`: bars are cut off at ±`zoom` percent, their labels still give the value."""
    _, fin = select(df, schedulers=schedulers)
    d = fin.group_by("case", "scheduler").agg(
        solved=(pl.col("cost") <= solved_cost).sum(), runs=pl.len()
    )
    base = d.filter(pl.col("scheduler") == BASELINE).select(
        "case", base="solved", base_runs="runs"
    )
    for case, b, _ in base.iter_rows():
        assert b > 0, f"{BASELINE} solved no runs of {case}"
    bases = ", ".join(f"{c} {b} of {r}" for c, b, r in base.sort("case").iter_rows())
    d = d.join(base, on="case").with_columns(
        gain=(pl.col("solved") / pl.col("base") - 1) * 100
    )
    # of lhs / rhs, all with the most proofs are top
    most = pl.col("solved").filter(DETOUR).max().over("case", "side")
    tops = d.with_columns(side=SIDE_OF).select(
        "case",
        "scheduler",
        "side",
        top=pl.when(DETOUR)
        .then(pl.col("solved") == most)
        .otherwise(pl.col("side") == "backoff"),
    )
    return (
        _bars(_titled(d, fin, tops), "gain", zoom=zoom)
        + p9.facet_wrap("title", nrow=1, scales="free")
        + p9.labs(
            x=f"more solved runs than {BASELINE} [%]",
            title="Proved goals",
            subtitle=f"runs that proved the goal, relative to {BASELINE}'s;\n"
            f"{BASELINE} solved {bases}; higher is better;\n"
            "opaque: the lhs / rhs with the most proofs of each case;\n"
            f"// = bar cut off; the line at 0 is {BASELINE}",
        )
    )


def _per_resource(df: pl.DataFrame, *cols: str) -> pl.DataFrame:
    """`df` once per resource, `x` its running max."""
    return pl.concat(
        df.select(
            *KEY,
            *cols,
            resource=pl.lit(label, RESOURCE),
            x=pl.col(r).cum_max().over(KEY),
        )
        for r, label in RESOURCES.items()
    )


def solved_profile(
    df: pl.DataFrame,
    schedulers: Sequence[Scheduler] = SCHEDULERS,
    *,
    solved_cost: float = 1,
    log_x: bool = False,
    y_min: float = 0,
    y_max: float = 1,
) -> p9.ggplot:
    """Fraction of runs that reached `solved_cost` within a resource budget.
    `log_x`: log x axis; `y_min`, `y_max`: ends of the y axis, to zoom in on a range."""
    df, fin = select(df, schedulers=schedulers)
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
        .join(
            proofs.group_by(group).agg(
                pl.col("y").max(), n=pl.len(), last=pl.col("x").max()
            ),
            on=group,
            how="left",
        )
        .with_columns(pl.col("y", "n").fill_null(0), side=SIDE_OF)
    )
    # step curves from 0 at `lo` through the proofs to `hi`
    curves = pl.concat(
        [
            ends.select(*group, x="lo", y=pl.lit(0.0)),
            proofs.select(*group, "x", "y"),
            ends.select(*group, x="hi", y="y"),
        ]
    ).sort("x", maintain_order=True)
    # backoff is always top; per resource, of lhs / rhs the one with the most proofs,
    # ties going to the smallest budget of the last proof, then to the smallest k
    tops = ends.sort(
        "n", "last", K_OF, descending=[True, False, False], nulls_last=True
    ).select(*group, "side", top=_top("case", "resource"))
    curves = _titled(curves, fin, tops)
    return (
        # curves that stay at 0 get no star
        _steps(curves, "y", ["case", "resource"], log_x, star_if=pl.col("y") > 0)
        # few proof cases, so the resources go side by side
        + _facets("resource", 1, "free_x")
        + p9.coord_cartesian(ylim=(y_min, y_max))
        + p9.labs(
            x="",
            y="runs with a proof",
            title="Proving the goal",
            subtitle="fraction of runs that proved the goal within a time / e-graph"
            " size budget; higher is better;\na curve ends at the most its scheduler"
            " used, ★ marks its last proof;\nopaque: backoff and the lhs / rhs with"
            " the most proofs, ties going to the smallest budget of the last one",
        )
    )


def _relative(
    df: pl.DataFrame, fin: pl.DataFrame, metric: Metric, absolute: bool = False
) -> pl.DataFrame:
    """`df` with `rel` = `metric` / the run's smallest final one (or `metric`)."""
    if absolute:
        return df.with_columns(rel=pl.col(metric))
    best = fin.group_by("case", "run").agg(best=pl.col(metric).min())
    return df.join(best, on=["case", "run"]).with_columns(
        rel=pl.col(metric) / pl.col("best")
    )


def _grid(d: pl.DataFrame, by: list[str], log: bool, n: int = 300) -> pl.DataFrame:
    """`n` evenly (or log-) spaced `x` per group of `by`."""
    lo, hi = pl.col("x").min(), pl.col("x").max()
    if log:
        lo, hi = lo.log(), hi.log()
    grid = d.group_by(by).agg(x=pl.linear_spaces(lo, hi, n)).explode("x")
    return grid.with_columns(pl.col("x").exp()) if log else grid


def _geomean_curves(d: pl.DataFrame, by: list[str], grid: pl.DataFrame) -> pl.DataFrame:
    """Geomean of `rel` at each `x` of `grid`, runs held at their ends."""
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
    """Cost relative to the run's best final cost over resource, geomean over runs.
    `log_x`/`log_y`: log axes; `absolute`: plain cost (only the y scale changes)."""
    df, fin = select(df, schedulers=schedulers)
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
    """Cost, time and size relative to the run's smallest final one, per iteration.
    `log_x`/`log_y`: log axes; `absolute`: plain values (only the y scale changes)."""
    df, fin = select(df, schedulers=schedulers)
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
        _steps(curves, "rel", by, log_x, star_if=pl.col("metric") == METRICS["cost"])
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
