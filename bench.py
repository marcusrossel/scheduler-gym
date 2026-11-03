#!/usr/bin/env python3

import asyncio
import csv
import os
import shutil
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
BENCHDATA = ROOT / "benchdata"
SYS_CPU = Path("/sys/devices/system/cpu")
# written by `benchmode.sh on`
BENCHMODE_CPUS = Path("/run/benchmode/bench-cpus")
# every scheduler in schedulers/, without the .rs suffix
SCHEDULERS = tuple(sorted(p.stem for p in (ROOT / "schedulers").glob("*.rs")))
CASE_STUDIES = ("caviar", "szalinski", "trig", "integ")


def physical_cpus() -> list[list[int]]:
    """The allowed CPUs, one per physical core, grouped by shared L3 cache."""
    groups: dict[str, list[int]] = {}
    seen_cores = set()
    for cpu in sorted(os.sched_getaffinity(0)):
        d = SYS_CPU / f"cpu{cpu}"
        core = (d / "topology/core_cpus_list").read_text().strip()
        if core in seen_cores:
            continue  # SMT sibling
        seen_cores.add(core)
        l3 = (d / "cache/index3/shared_cpu_list").read_text().strip()
        groups.setdefault(l3, []).append(cpu)
    return list(groups.values())


def assign_cpus() -> dict[str, list[int]]:
    """Gives each case study its own physical cores, never spanning two L3 caches."""
    groups = physical_cpus()
    assignment = {}
    for i, cpus in enumerate(groups):
        cases = CASE_STUDIES[i :: len(groups)]
        for j, case in enumerate(cases):
            lo, hi = len(cpus) * j // len(cases), len(cpus) * (j + 1) // len(cases)
            assignment[case] = cpus[lo:hi]
    if not all(assignment.get(case) for case in CASE_STUDIES):
        raise SystemExit(
            f"not enough cores for {len(CASE_STUDIES)} case studies: {groups}\n"
            "in bench mode, start this via `./benchmode.sh run uv run bench.py`"
        )
    return {case: assignment[case] for case in CASE_STUDIES}


def find_tctl() -> Path | None:
    for hwmon in Path("/sys/class/hwmon").iterdir():
        if (hwmon / "name").read_text().strip() == "k10temp":
            return hwmon / "temp1_input"
    return None


async def monitor(cpus: dict[str, list[int]], running: dict[str, str]):
    """Logs CPU temperature and per-case-study frequency, to check for throttling."""
    tctl = find_tctl()
    path = BENCHDATA / f"thermals-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    # min_mhz only gets entries for case studies that actually ran something
    max_temp, min_mhz = float("-inf"), {}
    with path.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            ["time", "tctl"] + [f"{c}_{x}" for c in cpus for x in ("scheduler", "mhz")]
        )
        try:
            while True:
                temp = int(tctl.read_text()) / 1000 if tctl else float("nan")
                max_temp = max(max_temp, temp)
                row = [f"{time.time():.1f}", f"{temp:.1f}"]
                for case, case_cpus in cpus.items():
                    mhz = (
                        sum(
                            int(
                                (
                                    SYS_CPU / f"cpu{c}/cpufreq/cpuinfo_avg_freq"
                                ).read_text()
                            )
                            for c in case_cpus
                        )
                        / len(case_cpus)
                        / 1000
                    )
                    if running[case]:
                        min_mhz[case] = min(min_mhz.get(case, mhz), mhz)
                    row += [running[case], f"{mhz:.0f}"]
                out.writerow(row)
                f.flush()
                await asyncio.sleep(1)
        finally:
            print(f"thermal log: {path}")
            temp = f"{max_temp:.1f} °C" if tctl else "n/a"
            print(
                f"max Tctl {temp}, min MHz while running { {c: round(m) for c, m in min_mhz.items()} }"
            )


async def bench_case_study(
    case: str, position: int, tmp: Path, cpus: list[int], running: dict[str, str]
):
    # per-case-study files, so case studies can run in parallel
    sched = tmp / f"{case}.scheduler.rs"
    entries = tmp / f"{case}.entries.txt"

    # schedulers run sequentially per case study, since each builds in place
    width = max(map(len, CASE_STUDIES))
    with tqdm(SCHEDULERS, desc=case.ljust(width), position=position) as bar:
        for scheduler in bar:
            bar.set_postfix_str(scheduler)

            out_dir = BENCHDATA / f"{scheduler}.rs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_entries = out_dir / f"{case}.entries"
            log = out_dir / f"{case}.log"
            if out_entries.exists():
                continue

            running[case] = scheduler
            entries.unlink(missing_ok=True)
            sched.write_text(
                (ROOT / "schedulers" / f"{scheduler}.rs").read_text()
                + (ROOT / "gym-common.rs").read_text()
            )

            with log.open("w") as f:
                # pinned to its own cores, so builds of one case study can't disturb another
                proc = await asyncio.create_subprocess_exec(
                    "taskset",
                    "-c",
                    ",".join(map(str, cpus)),
                    "bash",
                    "./run.sh",
                    str(sched),
                    cwd=ROOT / "case-studies" / case,
                    env={**os.environ, "GYM_ENTRIES": str(entries)},
                    stdout=f,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await proc.wait()

            if entries.exists():
                shutil.move(entries, out_entries)
            else:
                tqdm.write(
                    f"[{case}] scheduler '{scheduler}' FAILED (no entries), see {log}"
                )
        running[case] = ""
        bar.set_postfix_str("")


async def main():
    BENCHDATA.mkdir(exist_ok=True)
    if not BENCHMODE_CPUS.exists():
        print("WARNING: bench mode is off, results may suffer from boost/throttling")
    cpus = assign_cpus()
    for case, case_cpus in cpus.items():
        print(f"{case}: cpus {case_cpus}")

    running = dict.fromkeys(CASE_STUDIES, "")
    mon = asyncio.create_task(monitor(cpus, running))
    with tempfile.TemporaryDirectory() as tmp:
        async with asyncio.TaskGroup() as tg:
            for position, case in enumerate(CASE_STUDIES):
                tg.create_task(
                    bench_case_study(case, position, Path(tmp), cpus[case], running)
                )
    mon.cancel()
    # only swallow the cancellation, so a crashed monitor doesn't go unnoticed
    try:
        await mon
    except asyncio.CancelledError:
        pass

    print("FINISHED ALL")


if __name__ == "__main__":
    asyncio.run(main())
