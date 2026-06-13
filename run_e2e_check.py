import subprocess
import sys
import json
import os
from datetime import datetime

ROOT = os.path.dirname(__file__)
E2E = os.path.join(ROOT, "e2e_run.py")
REPORT = os.path.join(ROOT, "e2e_report.json")

def run_e2e(timeout=300):
    cmd = [sys.executable, E2E]
    env = os.environ.copy()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        return {
            "success": False,
            "reason": "timeout",
            "output": out,
        }
    # heuristics to detect failure
    lower = out.lower() if out else ""
    errors = []
    for token in ["error", "exception", "traceback", "failed"]:
        if token in lower:
            errors.append(token)
    success_marker = "e2e run complete." in lower
    success = proc.returncode == 0 and success_marker and not errors
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "returncode": proc.returncode,
        "success": success,
        "errors_detected": errors,
        "output": out,
    }
    return report


if __name__ == "__main__":
    print("Running e2e_run.py (this may take a while)...")
    rep = run_e2e()
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    if rep["success"]:
        print("E2E check PASSED")
    else:
        print("E2E check FAILED")
    print(f"Report written to: {REPORT}")
    # print short summary
    print(json.dumps({k: rep[k] for k in ("timestamp","returncode","success","errors_detected")}, indent=2))
    # exit non-zero on failure
    sys.exit(0 if rep["success"] else 2)
