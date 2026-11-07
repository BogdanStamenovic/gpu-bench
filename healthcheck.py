#!/usr/bin/env python3
"""GPU health check: VRAM integrity, verified compute burn, thermals/clocks under
sustained load, and error history. Writes results/<slug>.json and results/<slug>.md.

Run:  ./healthcheck.py            (full: ~15 min)
      ./healthcheck.py --quick    (smoke test: ~2 min)

Stdlib only. Needs nvidia-smi, nvcc (to build the two CUDA tools) and journalctl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"

TELEMETRY_FIELDS = [
    "timestamp", "temperature.gpu", "fan.speed", "power.draw", "enforced.power.limit",
    "clocks.gr", "clocks.mem", "utilization.gpu", "pstate", "clocks_event_reasons.active",
    "pcie.link.gen.gpucurrent", "pcie.link.width.current", "memory.used",
]
# bits of clocks_event_reasons.active (nvml.h NVML_CLOCKS_EVENT_REASON_*)
REASON_BITS = {
    0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
    0x100: "display_clocks",
}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def sh(cmd: list[str], timeout: float = 60) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


# ---------------------------------------------------------------- call safety
# This box takes phone calls on the same GPU (hotline-call + gpu-share's lease).
# A call must never be starved of VRAM, so every GPU-heavy step runs under a
# watchdog that stops it within ~1 s of a call starting. On machines without
# these services the probes fail closed to "no call" and the check runs normally.

def call_active() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8792/health", timeout=2) as r:
            h = json.load(r)
        if h.get("active_calls") or h.get("inbound", {}).get("inCall"):
            return True
    except Exception:
        pass
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/gpu-share/status", timeout=2) as r:
            s = json.load(r)
        if s.get("call_active") or s.get("call_leases"):
            return True
    except Exception:
        pass
    return False


def wait_call_free(max_wait_s: float = 3600) -> None:
    t0 = time.time()
    announced = False
    while call_active():
        if not announced:
            log("call in progress, waiting for it to end before using the GPU")
            announced = True
        if time.time() - t0 > max_wait_s:
            raise SystemExit("gave up waiting for the call to end")
        time.sleep(5)
    if announced:
        time.sleep(10)  # let the call's VRAM drain


def run_guarded(cmd: list[str], timeout: float, retries: int = 3) -> tuple[int, str, bool]:
    """Run a GPU-heavy tool; SIGTERM it if a call starts, then retry after the call."""
    for attempt in range(retries + 1):
        wait_call_free()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, text=True)
        stopped = threading.Event()

        def watch() -> None:
            while p.poll() is None:
                if call_active():
                    log("call started: stopping the test to free the GPU")
                    stopped.set()
                    p.send_signal(signal.SIGTERM)
                    return
                time.sleep(1)

        threading.Thread(target=watch, daemon=True).start()
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            return 124, out, False
        if not stopped.is_set():
            return p.returncode, out, False
        log(f"interrupted by a call (attempt {attempt + 1}); will retry after it ends")
    return p.returncode, out, True


# ---------------------------------------------------------------- telemetry

class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self.proc: subprocess.Popen | None = None
        self.marks: list[tuple[float, str]] = []

    def start(self) -> None:
        f = open(self.path, "w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=f, stderr=subprocess.DEVNULL)

        self.power: list[tuple[float, float, float]] = []   # (t, avg W, max W)
        self._stop = threading.Event()
        threading.Thread(target=self._power_loop, daemon=True).start()

    def _power_loop(self) -> None:
        # On this driver the power.draw query fields read N/A, but the
        # "Power Samples" block of `nvidia-smi -q -d POWER` is populated (a
        # rolling window of ~119 samples). Poll it every 2 s as the power source.
        while not self._stop.is_set():
            try:
                txt = sh(["nvidia-smi", "-q", "-d", "POWER"], timeout=10)
                blk = txt.split("Power Samples", 1)[1].splitlines()[:6]
                vals = {l.split(":")[0].strip(): l.split(":")[1].strip() for l in blk if ":" in l}
                avg, mx = num(vals.get("Avg", "").split(" ")[0]), num(vals.get("Max", "").split(" ")[0])
                if avg is not None:
                    self.power.append((time.time(), avg, mx or avg))
            except Exception:
                pass
            self._stop.wait(2)

    def mark(self, label: str) -> None:
        self.marks.append((time.time(), label))

    def stop(self) -> None:
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

    def rows(self) -> list[dict]:
        out = []
        for line in self.path.read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != len(TELEMETRY_FIELDS):
                continue
            r = dict(zip(TELEMETRY_FIELDS, parts))
            try:
                r["t"] = dt.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            out.append(r)
        return out


def num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict], t0: float, t1: float, power: list | None = None) -> dict:
    seg = [r for r in rows if t0 <= r["t"] <= t1]
    if not seg:
        return {}
    res: dict = {"samples": len(seg)}
    pw = [p for p in (power or []) if t0 + 3 <= p[0] <= t1]   # skip the window that straddles the phase start
    for key, name in [("temperature.gpu", "temp_c"), ("fan.speed", "fan_pct"),
                      ("power.draw", "power_w"), ("clocks.gr", "core_mhz"),
                      ("clocks.mem", "mem_mhz"), ("utilization.gpu", "util_pct")]:
        vals = [v for v in (num(r[key]) for r in seg) if v is not None]
        if vals:
            # steady state = the last 60% of the segment, after warm-up
            tail = vals[int(len(vals) * 0.4):] or vals
            res[name] = {"min": min(vals), "max": max(vals),
                         "steady_avg": round(sum(tail) / len(tail), 1)}
        else:
            res[name] = None   # e.g. power.draw is N/A on some boards/drivers
    if res.get("power_w") is None and pw:
        avgs = [p[1] for p in pw]
        tail = avgs[int(len(avgs) * 0.4):] or avgs
        res["power_w"] = {"min": min(avgs), "max": max(p[2] for p in pw),
                          "steady_avg": round(sum(tail) / len(tail), 1), "source": "nvidia-smi -q power samples"}
    reasons: dict[str, int] = {}
    for r in seg:
        try:
            bits = int(r["clocks_event_reasons.active"], 16)
        except ValueError:
            continue
        for b, n in REASON_BITS.items():
            if bits & b:
                reasons[n] = reasons.get(n, 0) + 1
    res["clock_event_reason_seconds"] = reasons
    res["pcie_gen_seen"] = sorted({r["pcie.link.gen.gpucurrent"] for r in seg})
    res["pcie_width_seen"] = sorted({r["pcie.link.width.current"] for r in seg})
    return res


# ---------------------------------------------------------------- facts

def gpu_identity() -> dict:
    q = ("name,vbios_version,driver_version,pci.bus_id,memory.total,power.default_limit,"
         "power.max_limit,clocks.max.gr,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max,"
         "temperature.gpu.tlimit")
    vals = [x.strip() for x in sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"]).split(",")]
    ident = dict(zip(q.split(","), vals))
    full = sh(["nvidia-smi", "-q"])
    for line in full.splitlines():
        s = line.strip()
        if s.startswith("Replays Since Reset"):
            ident["pcie_replays_since_reset"] = s.split(":")[1].strip()
        if s.startswith("Product Architecture"):
            ident["architecture"] = s.split(":")[1].strip()
    # Lifetime-of-driver counters for throttle time; deltas over the run matter.
    ident["event_counters_us"] = event_counters(full)
    return ident


def event_counters(full: str) -> dict:
    out, inside = {}, False
    for line in full.splitlines():
        if "Clocks Event Reasons Counters" in line:
            inside = True
            continue
        if inside:
            if ":" not in line:
                break
            k, v = [x.strip() for x in line.split(":", 1)]
            if not v.endswith("us"):
                break
            out[k] = int(v.split()[0])
    return out


def journal_errors(since: str | None = None) -> dict:
    cmd = ["journalctl", "-k", "--no-pager", "-o", "short-iso"]
    if since:
        cmd += ["--since", since]
    else:
        cmd += ["--since", "1970-01-01"]  # every boot the journal kept
    text = sh(cmd, timeout=120)
    xid = [l for l in text.splitlines() if "NVRM: Xid" in l]
    # AER lines that name the GPU's bus address (setup lines like "AER: enabled" are not errors)
    bus = sh(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]).strip().lower()
    short = bus[-12:] if bus else "nomatch"   # 0000:05:00.0 form
    aer = [l for l in text.splitlines() if "AER" in l and short in l.lower() and "enabled" not in l]
    boot_lines = [l for l in sh(["journalctl", "--list-boots", "--no-pager"]).splitlines()[1:] if l.strip()]
    boots = len(boot_lines)
    first = " ".join(boot_lines[0].split()[2:6]) if boot_lines else None
    return {"xid_count": len(xid), "xid_lines": xid[-20:], "aer_count": len(aer),
            "aer_lines": aer[-20:], "journal_first_entry": first, "boots_in_journal": boots}


def host_info() -> dict:
    cuda = sh(["nvcc", "--version"]).strip().splitlines()[-1] if shutil_which("nvcc") else None
    return {"hostname": platform.node(), "kernel": platform.release(),
            "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
            "nvcc": cuda}


def shutil_which(x: str) -> str | None:
    from shutil import which
    return which(x)


def build_tools() -> None:
    BUILD.mkdir(exist_ok=True)
    gencode = ["-gencode", "arch=compute_86,code=sm_86", "-gencode", "arch=compute_89,code=sm_89",
               "-gencode", "arch=compute_89,code=compute_89"]  # PTX fallback for newer cards
    for name, extra in [("memtest", []), ("burn", ["-lcublas"])]:
        src, exe = ROOT / "src" / f"{name}.cu", BUILD / name
        if exe.exists() and exe.stat().st_mtime > src.stat().st_mtime:
            continue
        log(f"building {name}")
        subprocess.run(["nvcc", "-O3", *gencode, "-o", str(exe), str(src), *extra], check=True)


# ---------------------------------------------------------------- verdict

def verdict(r: dict) -> tuple[str, list[str], list[str]]:
    fails, warns = [], []
    mt = r["vram_test"]
    if mt.get("errors", 0) > 0:
        fails.append(f"VRAM test: {mt['errors']} bad words ({mt['bit_flips']} bit flips)")
    if mt.get("interrupted"):
        warns.append("VRAM test was interrupted and did not finish")
    for b in r["burn"]:
        if b.get("mismatched_elements", 0) > 0:
            fails.append(f"burn {b['mode']}: {b['mismatched_elements']} wrong results in {b['mismatched_checks']} checks")
        if b.get("interrupted"):
            warns.append(f"burn {b['mode']} was interrupted")
        s = b.get("tflops_series") or []
        if len(s) >= 4:
            first, last = s[1][1], s[-1][1]   # skip the warm-up sample
            if last < first * 0.85:
                warns.append(f"burn {b['mode']}: throughput fell {100 * (1 - last / first):.0f}% over the run")
    je = r["journal_during_run"]
    if je["xid_count"]:
        fails.append(f"{je['xid_count']} Xid error(s) logged during the run")
    if je["aer_count"]:
        warns.append(f"{je['aer_count']} PCIe AER message(s) for the GPU during the run")
    jh = r["journal_history"]
    if jh["xid_count"]:
        warns.append(f"{jh['xid_count']} Xid error(s) in journal history (see xid_lines)")
    d = r["event_counter_delta_us"]
    for k in ("HW Thermal Slowdown", "HW Power Braking"):
        if d.get(k, 0) > 0:
            fails.append(f"{k} active for {d[k] / 1e6:.1f} s during the run (hardware protection kicked in)")
    if d.get("SW Thermal Slowdown", 0) > 0:
        warns.append(f"SW thermal slowdown for {d['SW Thermal Slowdown'] / 1e6:.1f} s (running at its temperature limit)")
    load = r["telemetry"].get("load", {})
    t = (load.get("temp_c") or {}).get("max")
    if t is not None and t >= 87:
        warns.append(f"peak temperature {t:.0f} C under load")
    w_max = r["identity"].get("pcie.link.width.max")
    if w_max and load.get("pcie_width_seen") and max(load["pcie_width_seen"], key=int) != w_max:
        fails.append(f"PCIe link ran at x{max(load['pcie_width_seen'])} (card supports x{w_max})")
    g_max = r["identity"].get("pcie.link.gen.max")
    if g_max and load.get("pcie_gen_seen") and str(max(int(x) for x in load["pcie_gen_seen"])) != g_max:
        warns.append(f"PCIe ran at Gen{max(load['pcie_gen_seen'])} under load (card supports Gen{g_max}; may be the slot)")
    rp = r.get("pcie_replays_delta")
    if rp:
        warns.append(f"{rp} PCIe replays during the run (signal integrity / riser / slot)")
    status = "FAIL" if fails else ("PASS with notes" if warns else "PASS")
    return status, fails, warns


# ---------------------------------------------------------------- report

def render_md(r: dict) -> str:
    i, tl = r["identity"], r["telemetry"]
    L = []
    L.append(f"# GPU health check: {i['name']}  ->  **{r['verdict']}**\n")
    L.append(f"Run {r['started']} on {r['host']['hostname']} ({r['mode']} mode, {r['duration_s'] / 60:.1f} min).\n")
    if r["fails"]:
        L += ["**Failures**", *[f"- {x}" for x in r["fails"]], ""]
    if r["warns"]:
        L += ["**Notes**", *[f"- {x}" for x in r["warns"]], ""]
    L.append("## Card\n")
    L.append("| | |\n|---|---|")
    for k, lab in [("name", "Model"), ("architecture", "Architecture"), ("vbios_version", "VBIOS"),
                   ("memory.total", "VRAM"), ("power.default_limit", "Power limit (default)"),
                   ("power.max_limit", "Power limit (max)"), ("clocks.max.gr", "Max core clock"),
                   ("clocks.max.mem", "Max memory clock"), ("pcie.link.gen.max", "PCIe gen (max)"),
                   ("pcie.link.width.max", "PCIe width (max)"), ("driver_version", "Driver")]:
        L.append(f"| {lab} | {i.get(k, '?')} |")
    mt = r["vram_test"]
    L.append("\n## VRAM integrity\n")
    L.append(f"{mt.get('tested_mib')} of {mt.get('total_mib')} MiB tested, {mt.get('passes_requested')} passes of "
             f"{len(mt.get('tests', [])) // max(mt.get('passes_requested', 1), 1)} patterns, {mt.get('seconds')} s: "
             f"**{mt.get('errors')} errors, {mt.get('bit_flips')} bit flips**. "
             f"Pattern write+read bandwidth: {r['vram_bandwidth_gbps']} GB/s.\n")
    L.append("## Sustained load (verified compute burn)\n")
    L.append("| Phase | Time | Wrong results | TFLOPS avg (min-max sample) |\n|---|---:|---:|---|")
    for b in r["burn"]:
        L.append(f"| {b['mode']} {'tensor cores' if b['mode'] == 'fp16' else 'CUDA cores'} | {b['seconds']:.0f} s | "
                 f"{b['mismatched_elements']} | {b['tflops_avg']} ({b['tflops_min_sample']}-{b['tflops_max_sample']}) |")
    L.append("\n| Under load | min | max | steady avg |\n|---|---:|---:|---:|")
    load = tl.get("load", {})
    for k, lab in [("temp_c", "Temperature (C)"), ("fan_pct", "Fan (%)"), ("power_w", "Power (W)"),
                   ("core_mhz", "Core clock (MHz)"), ("mem_mhz", "Memory clock (MHz)"), ("util_pct", "Utilization (%)")]:
        v = load.get(k)
        L.append(f"| {lab} | {v['min']:.0f} | {v['max']:.0f} | {v['steady_avg']:.0f} |" if v else f"| {lab} | n/a | n/a | n/a |")
    L.append(f"\nClock limiters active (seconds of samples): {load.get('clock_event_reason_seconds') or 'none'}. "
             f"PCIe under load: Gen{'/'.join(load.get('pcie_gen_seen', []))} x{'/'.join(load.get('pcie_width_seen', []))}.\n")
    cd = tl.get("cooldown", {})
    if cd.get("temp_c"):
        L.append(f"Cool-down ({r['cooldown_s']} s after load): {cd['temp_c']['max']:.0f} C -> {cd['temp_c']['min']:.0f} C.\n")
    ic = tl.get("idle", {})
    if ic.get("temp_c"):
        L.append(f"Idle before the run: {ic['temp_c']['steady_avg']:.0f} C, fan {ic['fan_pct']['steady_avg'] if ic.get('fan_pct') else 'n/a'} %.\n")
    L.append("## Error history\n")
    jh, jr = r["journal_history"], r["journal_during_run"]
    L.append(f"- Xid errors in the kernel journal: **{jh['xid_count']}** (journal covers {jh['boots_in_journal']} boots, "
             f"from {jh['journal_first_entry']}). During this run: **{jr['xid_count']}**.")
    L.append(f"- PCIe AER errors for the GPU: {jh['aer_count']} in history, {jr['aer_count']} during the run.")
    L.append(f"- PCIe replays: {i.get('pcie_replays_since_reset')} since driver load, {r.get('pcie_replays_delta')} during the run.")
    d = r["event_counter_delta_us"]
    L.append(f"- Throttle time during the run: " + ", ".join(f"{k} {v / 1e6:.1f} s" for k, v in d.items()) + "\n")
    L.append("## What this does and does not show\n")
    L += [
        "- It exercises the CUDA cores, tensor cores, VRAM and power delivery at full load, and checks every result.",
        "- It does **not** run the graphics pipeline (rasterizer, RT cores), display outputs, or video encode/decode.",
        "  Display outputs are untested because the test machine runs headless.",
        "- It cannot see the card's history before the journal starts, mining use, or fan bearing wear beyond what shows in temperatures.",
        "- A pass is evidence of health under this load for this duration, not a guarantee.",
    ]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="short smoke-test durations")
    ap.add_argument("--burn-s", type=float, help="seconds per burn phase (default 300, quick 20)")
    ap.add_argument("--mem-passes", type=int, help="VRAM test passes (default 300, quick 5)")
    ap.add_argument("--tag", default="", help="suffix for the result file names")
    a = ap.parse_args()
    burn_s = a.burn_s or (20 if a.quick else 300)
    passes = a.mem_passes or (5 if a.quick else 300)
    idle_s, cool_s = (5, 10) if a.quick else (30, 90)

    build_tools()
    RESULTS.mkdir(exist_ok=True)
    started = dt.datetime.now().astimezone()
    ident = gpu_identity()
    slug = f"{ident['name'].replace('NVIDIA GeForce ', '').replace(' ', '')}-{started:%Y%m%d-%H%M}{('-' + a.tag) if a.tag else ''}"
    log(f"health check of {ident['name']} -> results/{slug}.*")

    hist = journal_errors()
    log(f"journal: {hist['xid_count']} Xid, {hist['aer_count']} AER for the GPU, {hist['boots_in_journal']} boots kept")

    wait_call_free()
    tele = Telemetry(RESULTS / f"{slug}.telemetry.csv")
    tele.start()
    t_idle = time.time()
    time.sleep(idle_s)
    t_mem = time.time()

    log(f"VRAM test: {passes} passes")
    rc, out, _ = run_guarded([str(BUILD / "memtest"), "--passes", str(passes), "--reserve-mib", "400"],
                             timeout=passes * 20 + 120)
    vram = json.loads(out) if out.strip().startswith("{") else {"error": f"exit {rc}", "errors": 0}
    vram_bw = None
    tests = vram.get("tests", [])
    plain = [t for t in tests if "moving" not in t["test"]]
    if plain and vram.get("tested_mib"):
        vram_bw = round(sum(vram["tested_mib"] * 2 * 1048576 / t["seconds"] / 1e9 for t in plain) / len(plain), 1)
    log(f"VRAM test: {vram.get('errors')} errors over {vram.get('tested_mib')} MiB")

    t_load = time.time()
    burns = []
    for mode in ("fp32", "fp16"):
        log(f"burn {mode}: {burn_s:.0f} s")
        rc, out, _ = run_guarded([str(BUILD / "burn"), "--mode", mode, "--seconds", str(burn_s),
                                  "--sample-s", "10" if burn_s >= 60 else "5"], timeout=burn_s + 120)
        b = json.loads(out) if out.strip().startswith("{") else {"mode": mode, "error": f"exit {rc}",
                                                               "mismatched_elements": 0, "mismatched_checks": 0,
                                                               "seconds": 0, "tflops_avg": 0, "tflops_min_sample": 0,
                                                               "tflops_max_sample": 0}
        burns.append(b)
        log(f"burn {mode}: {b.get('tflops_avg')} TFLOPS, {b.get('mismatched_elements')} wrong results")
    t_cool = time.time()
    log(f"cool-down: {cool_s} s")
    time.sleep(cool_s)
    t_end = time.time()
    tele.stop()

    ident_after = gpu_identity()
    c0, c1 = ident["event_counters_us"], ident_after["event_counters_us"]
    rows = tele.rows()
    try:
        replays = int(ident_after.get("pcie_replays_since_reset", 0)) - int(ident.get("pcie_replays_since_reset", 0))
    except ValueError:
        replays = None
    r = {
        "schema": 1, "mode": "quick" if a.quick else "full", "started": started.isoformat(timespec="seconds"),
        "duration_s": round(t_end - t_idle), "host": host_info(), "identity": ident,
        "journal_history": hist,
        "journal_during_run": journal_errors(since=started.strftime("%Y-%m-%d %H:%M:%S")),
        "vram_test": vram, "vram_bandwidth_gbps": vram_bw, "burn": burns,
        "telemetry": {"idle": summarize(rows, t_idle, t_mem, tele.power),
                      "vram_test": summarize(rows, t_mem, t_load, tele.power),
                      "load": summarize(rows, t_load, t_cool, tele.power),
                      "cooldown": summarize(rows, t_cool, t_end, tele.power)},
        "power_samples": [[round(t - t_idle, 1), a, m] for t, a, m in tele.power],
        "cooldown_s": cool_s,
        "event_counter_delta_us": {k: c1.get(k, 0) - c0.get(k, 0) for k in c1},
        "pcie_replays_delta": replays,
    }
    r["verdict"], r["fails"], r["warns"] = verdict(r)
    (RESULTS / f"{slug}.json").write_text(json.dumps(r, indent=2) + "\n")
    (RESULTS / f"{slug}.md").write_text(render_md(r))
    log(f"verdict: {r['verdict']}  ->  results/{slug}.md")
    print(RESULTS / f"{slug}.md")
    return 1 if r["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
