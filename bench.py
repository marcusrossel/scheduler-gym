#!/usr/bin/env python3

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from tqdm import tqdm

from gym import CASE_STUDIES, SCHEDULERS

ROOT = Path(__file__).resolve().parent
BENCHDATA = ROOT / "benchdata"


async def bench_case_study(case: str, position: int, tmp: Path):
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

            entries.unlink(missing_ok=True)
            sched.write_text(
                (ROOT / "schedulers" / f"{scheduler}.rs").read_text()
                + (ROOT / "gym-common.rs").read_text()
            )

            with log.open("w") as f:
                proc = await asyncio.create_subprocess_exec(
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
        bar.set_postfix_str("")


async def main():
    BENCHDATA.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        async with asyncio.TaskGroup() as tg:
            for position, case in enumerate(CASE_STUDIES):
                tg.create_task(bench_case_study(case, position, Path(tmp)))

    print("FINISHED ALL")


if __name__ == "__main__":
    asyncio.run(main())
