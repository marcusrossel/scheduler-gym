#!/usr/bin/env bash
# Switch the machine into a reproducible benchmarking mode and back.
#
#   sudo ./benchmode.sh on       disable boost + SMT, lock the frequency,
#                                confine system/user processes to the housekeeping CPU
#   sudo ./benchmode.sh off      restore the settings saved by `on`
#   ./benchmode.sh run CMD...    run CMD (as you) on the benchmark CPUs, e.g.
#                                ./benchmode.sh run uv run bench.py
#   ./benchmode.sh status        show the current settings
#
# Everything is runtime-only, a reboot also restores the normal settings.
# Environment: BENCH_FREQ_MHZ (default: nominal/base frequency), HOUSEKEEPING_CPU (default: 0)

set -euo pipefail

CPU=/sys/devices/system/cpu
STATE_DIR=/run/benchmode
STATE="$STATE_DIR/sysfs"
SLICES_STATE="$STATE_DIR/slices"
CPUS_STATE="$STATE_DIR/bench-cpus"
SLICES=(system.slice user.slice init.scope)
HOUSEKEEPING_CPU="${HOUSEKEEPING_CPU:-0}"

die() { echo "benchmode: $*" >&2; exit 1; }
need_root() { [[ $EUID -eq 0 ]] || die "'$1' needs root, use sudo"; }

# write a sysfs value, warn instead of aborting on failure
w() { echo "$2" > "$1" 2>/dev/null || echo "benchmode: could not write '$2' to $1" >&2; }

cpufreq_files() {
    local c
    for c in "$CPU"/cpu[0-9]*/cpufreq; do
        # restore order matters: governor before EPP, max before min
        echo "$c/scaling_governor"
        if [[ -e $c/energy_performance_preference ]]; then echo "$c/energy_performance_preference"; fi
        echo "$c/scaling_max_freq"
        echo "$c/scaling_min_freq"
    done
}

online_cpus() {
    local c n
    for c in "$CPU"/cpu[0-9]*; do
        n="${c##*cpu}"
        if [[ ! -e $c/online || $(<"$c/online") == 1 ]]; then echo "$n"; fi
    done | sort -n
}

cmd_on() {
    need_root on
    [[ -e $STATE ]] && die "bench mode is already on (run 'off' first)"

    local nominal="$CPU/cpu0/acpi_cppc/nominal_freq"
    local freq_mhz="${BENCH_FREQ_MHZ:-}"
    if [[ -z $freq_mhz ]]; then
        [[ -e $nominal ]] || die "$nominal not found, set BENCH_FREQ_MHZ"
        freq_mhz="$(<"$nominal")"
    fi
    local freq_khz=$((freq_mhz * 1000))

    mkdir -p "$STATE_DIR"

    # save everything before changing anything, while all CPUs are still online
    {
        # restore order matters: SMT and boost before the per-CPU frequencies
        for f in "$CPU/smt/control" "$CPU/cpufreq/boost" $(cpufreq_files); do
            if [[ -e $f ]]; then echo "$f $(<"$f")"; fi
        done
    } > "$STATE.tmp"
    for s in "${SLICES[@]}"; do
        echo "$s $(systemctl show -P AllowedCPUs "$s")"
    done > "$SLICES_STATE"
    mv "$STATE.tmp" "$STATE"

    w "$CPU/smt/control" off
    w "$CPU/cpufreq/boost" 0
    for c in $(online_cpus); do
        local p="$CPU/cpu$c/cpufreq"
        # with amd-pstate-epp the performance governor also forces EPP=performance
        w "$p/scaling_governor" performance
        w "$p/scaling_max_freq" "$freq_khz"
        w "$p/scaling_min_freq" "$freq_khz"
    done

    local bench_cpus
    bench_cpus="$(online_cpus | grep -vx "$HOUSEKEEPING_CPU" | paste -sd,)"
    echo "$bench_cpus" > "$CPUS_STATE"
    chmod 644 "$STATE" "$SLICES_STATE" "$CPUS_STATE"
    for s in "${SLICES[@]}"; do
        systemctl set-property --runtime "$s" AllowedCPUs="$HOUSEKEEPING_CPU"
    done

    echo "bench mode on: ${freq_mhz} MHz, housekeeping CPU $HOUSEKEEPING_CPU, bench CPUs $bench_cpus"
}

cmd_off() {
    need_root off
    [[ -e $STATE ]] || die "bench mode is not on"

    while read -r s cpus; do
        systemctl set-property --runtime "$s" AllowedCPUs="$cpus"
    done < "$SLICES_STATE"
    while read -r f v; do
        w "$f" "$v"
    done < "$STATE"

    rm -rf "$STATE_DIR"
    echo "bench mode off"
}

cmd_run() {
    [[ $EUID -ne 0 ]] || die "'run' should be called as your normal user (it uses sudo itself)"
    [[ -e $CPUS_STATE ]] || die "bench mode is not on"
    [[ $# -gt 0 ]] || die "usage: $0 run CMD..."

    # user.slice is confined to the housekeeping CPU, so the benchmark runs in its own
    # slice; the full environment is passed through so nix/direnv/uv setups keep working
    local envs
    mapfile -d '' envs < <(env -0)
    exec sudo systemd-run --quiet --pty --wait --collect --same-dir \
        --slice=bench.slice --uid="$(id -u)" --gid="$(id -g)" \
        -p AllowedCPUs="$(<"$CPUS_STATE")" \
        -- "$(command -v env)" -i "${envs[@]}" "$@"
}

cmd_status() {
    echo "bench mode:  $([[ -e $STATE ]] && echo on || echo off)"
    echo "smt:         $(<"$CPU/smt/control")"
    echo "boost:       $(<"$CPU/cpufreq/boost")"
    local p="$CPU/cpu1/cpufreq"
    echo "cpu1:        $(<"$p/scaling_governor"), $(($(<"$p/scaling_min_freq") / 1000))-$(($(<"$p/scaling_max_freq") / 1000)) MHz"
    for s in "${SLICES[@]}"; do
        printf '%-14s AllowedCPUs=%s\n' "$s:" "$(systemctl show -P AllowedCPUs "$s")"
    done
    if [[ -e $CPUS_STATE ]]; then echo "bench CPUs:  $(<"$CPUS_STATE")"; fi
}

case "${1:-}" in
    on) cmd_on ;;
    off) cmd_off ;;
    run) shift; cmd_run "$@" ;;
    status) cmd_status ;;
    *) die "usage: $0 on|off|run CMD...|status" ;;
esac
