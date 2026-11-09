"""Log GPU readings every 5 s and thermal zones every 10 s, forcing each line to disk so a sudden power-off keeps them.

    python -I gpu_log.py out.csv [hours]
"""
import csv, datetime, io, os, subprocess, sys, time

FIELDS = ("temperature.gpu,power.draw,clocks.gr,utilization.gpu,memory.used,pstate,clocks_throttle_reasons.active,"
          "clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,"
          "clocks_throttle_reasons.hw_power_brake_slowdown")
NO_WINDOW = 0x08000000


def run(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"ERR {type(e).__name__}"


def thermal():
    out = run(["typeperf", r"\Thermal Zone Information(*)\Temperature", r"\Processor Information(_Total)\% Processor Utility",
               "-sc", "1"], timeout=20)
    rows = [r for r in csv.reader(io.StringIO(out)) if len(r) > 1]
    if len(rows) < 2:
        return "thermal unavailable"
    return " ".join(f"{h.split(chr(92))[-2]}={v.strip()}" for h, v in zip(rows[0][1:], rows[1][1:]))


path, hours = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6
end = time.time() + hours * 3600
last_thermal, tz = 0, ""
with open(path, "a", encoding="utf-8", buffering=1) as f:
    f.write(f"# started {datetime.datetime.now():%F %T}; local time, {FIELDS}, then thermal zones in kelvin and CPU utility\n")
    while time.time() < end:
        if time.time() - last_thermal >= 10:
            tz, last_thermal = thermal(), time.time()
        gpu = run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader"])
        f.write(f"{datetime.datetime.now():%H:%M:%S}, {gpu}, {tz}\n")
        f.flush()
        os.fsync(f.fileno())
        tz = ""
        time.sleep(5)
