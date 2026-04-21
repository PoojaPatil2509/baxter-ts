"""
baxter-ts Pre-Launch Verification Script
=========================================
Run this on your machine BEFORE publishing to PyPI.
Checks every requirement for a production-ready library.

Usage:
    python pre_launch_check.py

Expected output: all items marked PASS
"""

import os
import re
import sys
import subprocess
import importlib
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")

# ── Console colours (work on Windows 10+ with ANSI enabled) ──────────
RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"

# Enable ANSI on Windows
if sys.platform == "win32":
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

passed = 0
failed = 0
warn_count = 0


def check(label, condition, fix="", warn=False):
    global passed, failed, warn_count
    if condition:
        print(f"  {GREEN}PASS{RESET}  {label}")
        passed += 1
    elif warn:
        print(f"  {YELLOW}WARN{RESET}  {label}")
        if fix:
            print(f"         Fix: {fix}")
        warn_count += 1
    else:
        print(f"  {RED}FAIL{RESET}  {label}")
        if fix:
            print(f"         Fix: {fix}")
        failed += 1


def section(title):
    print(f"\n{BOLD}{'─' * 55}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─' * 55}{RESET}")


# ══════════════════════════════════════════════════════════════════════
# DATA HELPERS — defined at top so they are always available
# ══════════════════════════════════════════════════════════════════════

def _make_base(n, seed=42):
    import numpy as np
    np.random.seed(seed)
    return np.linspace(100, 200, n) + np.random.randn(n) * 5


def _mk_missing(n):
    import numpy as np
    np.random.seed(42)
    v = _make_base(n)
    v[np.random.choice(n, int(n * 0.30), replace=False)] = float("nan")
    return v


def _mk_outliers(n):
    import numpy as np
    v = _make_base(n)
    for idx in [20, 80, 150, 250]:
        if idx < n:
            v[idx] = v[idx] * 15
    return v


def _mk_negative(n):
    import numpy as np
    np.random.seed(42)
    return np.linspace(-50, 50, n) + np.random.randn(n) * 3


def _read_html_safe(path):
    """
    Read HTML file always as UTF-8.
    The Plotly JS bundle contains characters outside Windows cp1252,
    so we MUST specify encoding='utf-8' explicitly — never rely on
    the system default (which is cp1252 on Windows).
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════

print(f"\n{BOLD}baxter-ts Pre-Launch Verification{RESET}")
print("=" * 55)

# ── 1. ENVIRONMENT ────────────────────────────────────────────────────
section("1. Environment")

check(
    f"Python >= 3.9  (found {sys.version.split()[0]})",
    sys.version_info >= (3, 9),
    fix="Install Python 3.9+",
)

REQUIRED_PKGS = [
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("sklearn",     "scikit-learn"),
    ("xgboost",     "xgboost"),
    ("catboost",    "catboost"),
    ("shap",        "shap"),
    ("plotly",      "plotly"),
    ("statsmodels", "statsmodels"),
    ("scipy",       "scipy"),
    ("jinja2",      "jinja2"),
]
for import_name, install_name in REQUIRED_PKGS:
    try:
        importlib.import_module(import_name)
        check(f"Package installed: {install_name}", True)
    except ImportError:
        check(f"Package installed: {install_name}", False,
              fix=f"pip install {install_name}")

# ── 2. IMPORTS ────────────────────────────────────────────────────────
section("2. Library imports")

bax_ok = False
try:
    from baxter_ts import BAXModel
    check("from baxter_ts import BAXModel", True)
    bax_ok = True
except Exception as e:
    check("from baxter_ts import BAXModel", False,
          fix=f"Run: pip install -e .    Error: {e}")

try:
    import baxter_ts
    ver = baxter_ts.__version__
    check(f"__version__ defined: {ver}", True)
except Exception:
    check("__version__ defined", False,
          fix="Add __version__ = '0.1.0' to baxter_ts/__init__.py")

# ── 3. CORE PIPELINE ─────────────────────────────────────────────────
section("3. Core pipeline (daily sales data)")

if not bax_ok:
    check("Core pipeline skipped — import failed", False)
else:
    try:
        import numpy as np
        import pandas as pd
        from baxter_ts import BAXModel

        np.random.seed(42)
        n = 400
        dates  = pd.date_range("2022-01-01", periods=n, freq="D")
        values = (
            np.linspace(100, 200, n)
            + 20 * np.sin(2 * np.pi * np.arange(n) / 7)
            + np.random.randn(n) * 5
        )
        values[np.random.choice(n, 15, replace=False)] = np.nan
        df = pd.DataFrame({"date": dates, "sales": values})

        model = BAXModel(n_cv_splits=2, verbose=False)
        model.fit(df, target_col="sales", date_col="date")
        check("model.fit() completes", True)

        forecast = model.predict(steps=14)
        check("model.predict() returns DataFrame",
              isinstance(forecast, pd.DataFrame))
        check("Forecast has 14 rows", len(forecast) == 14)
        check("Forecast values vary (not flat)",
              forecast["forecast"].nunique() > 1)

        anom = model.anomalies()
        check("model.anomalies() returns DataFrame",
              isinstance(anom, pd.DataFrame))
        check("anomaly_flag column present", "anomaly_flag" in anom.columns)

        narrative = model.explain()
        check("model.explain() returns non-empty string",
              isinstance(narrative, str) and len(narrative) > 50)

        sb = model.scoreboard()
        check("model.scoreboard() has 3 models", len(sb) == 3)

        sb_reset = sb.reset_index()
        winner_name = model._selector.best_model.name
        winner_rows = sb_reset[sb_reset["model"] == winner_name]
        winner_score = float(winner_rows["composite_score"].values[0])
        min_score    = float(sb_reset["composite_score"].min())
        check("Winner has lowest composite score",
              abs(winner_score - min_score) < 1e-9)

        summary = model.summary()
        required_keys = ["best_model", "test_mae", "test_rmse", "test_r2", "frequency"]
        check("model.summary() has all required keys",
              all(k in summary for k in required_keys))
        check(f"Best model: {summary['best_model']}", True)
        check(f"Test R²: {summary['test_r2']}", summary["test_r2"] is not None)

    except Exception as e:
        check("Core pipeline", False, fix=f"{e}\n{traceback.format_exc()}")

# ── 4. MULTI-FREQUENCY ────────────────────────────────────────────────
section("4. Multi-frequency support")

FREQ_TESTS = {
    "Hourly  (h)":  (pd.date_range("2023-01-01", periods=500, freq="h"),  "kwh"),
    "Daily   (D)":  (pd.date_range("2022-01-01", periods=365, freq="D"),  "sales"),
    "Weekly  (W)":  (pd.date_range("2021-01-04", periods=104, freq="W"),  "units"),
    "Monthly (MS)": (pd.date_range("2019-01-01", periods=60,  freq="MS"), "revenue"),
}

for freq_label, (dates, col) in FREQ_TESTS.items():
    try:
        import numpy as np
        import pandas as pd
        from baxter_ts import BAXModel

        np.random.seed(7)
        values = np.linspace(100, 200, len(dates)) + np.random.randn(len(dates)) * 5
        df_f   = pd.DataFrame({"date": dates, col: values})
        m_f    = BAXModel(n_cv_splits=2, verbose=False)
        m_f.fit(df_f, target_col=col, date_col="date")
        m_f.predict(steps=5)
        check(f"Frequency: {freq_label}", True)
    except Exception as e:
        check(f"Frequency: {freq_label}", False, fix=str(e))

# ── 5. HTML REPORT QUALITY ────────────────────────────────────────────
section("5. HTML report quality")

try:
    import numpy as np
    import pandas as pd
    from baxter_ts import BAXModel

    np.random.seed(42)
    n   = 400
    df_r = pd.DataFrame({
        "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
        "sales": np.linspace(100, 200, n) + np.random.randn(n) * 5,
    })
    m_r = BAXModel(n_cv_splits=2, verbose=False)
    m_r.fit(df_r, "sales", "date")
    m_r.predict(14)
    m_r.anomalies()

    # tempfile gives a safe cross-platform path without extension
    tmp_base = tempfile.mktemp(suffix="")
    report_path = m_r.report(tmp_base)
    check("report() creates .html file", os.path.exists(report_path))

    # ── FIX: always read as UTF-8, never system default ──────────────
    # Windows default encoding is cp1252 which cannot decode the
    # Plotly JS bundle (contains 0x8d and other non-cp1252 bytes).
    html = _read_html_safe(report_path)
    check("Report readable as UTF-8 (no encoding error)", True)

    # No CDN network calls
    cdn_tags = re.findall(r'<script[^>]+src=["\']https?://', html)
    check(
        "Zero CDN script tags — fully offline capable",
        len(cdn_tags) == 0,
        fix="generator.py must use _get_plotly_js_tag() to embed JS inline",
    )

    # One inline plotly bundle
    inline_cnt = len(re.findall(r'<script type=["\']text/javascript["\']>', html))
    check(
        f"Plotly JS embedded inline (found {inline_cnt}, need 1)",
        inline_cnt == 1,
        fix="_get_plotly_js_tag() in generator.py must embed plotly.min.js",
    )

    # Charts rendered
    chart_cnt = len(re.findall(r"Plotly\.newPlot", html))
    check(
        f"{chart_cnt} Plotly.newPlot calls found (need >= 5)",
        chart_cnt >= 5,
    )

    # File size sanity
    size_mb = os.path.getsize(report_path) / 1024 / 1024
    check(
        f"Report file size: {size_mb:.1f} MB (expect 4–6 MB with inline JS)",
        1 < size_mb < 10,
    )

    os.unlink(report_path)

except Exception as e:
    check("Report generation", False, fix=f"{e}\n{traceback.format_exc()}")

# ── 6. DATA QUALITY ROBUSTNESS ────────────────────────────────────────
section("6. Data quality robustness")

ROBUSTNESS_CASES = [
    ("30% missing values",      _mk_missing),
    ("Extreme outlier spikes",  _mk_outliers),
    ("Negative values",         _mk_negative),
]

for case_name, fn in ROBUSTNESS_CASES:
    try:
        import numpy as np
        import pandas as pd
        from baxter_ts import BAXModel

        np.random.seed(42)
        n    = 350
        df_c = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
            "value": fn(n),
        })
        m_c = BAXModel(n_cv_splits=2, verbose=False)
        m_c.fit(df_c, "value", "date")
        m_c.predict(10)
        m_c.anomalies()
        check(f"Handles: {case_name}", True)
    except Exception as e:
        check(f"Handles: {case_name}", False, fix=str(e))

# ── 7. BUILD ARTIFACTS ────────────────────────────────────────────────
section("7. Build artifacts")

check("dist/ folder exists",
      os.path.isdir("dist"),
      fix="Run: python -m build")

whl_files = [f for f in os.listdir("dist") if f.endswith(".whl")] \
    if os.path.isdir("dist") else []
check(
    f"Wheel (.whl) exists: {whl_files[0] if whl_files else 'NOT FOUND'}",
    len(whl_files) > 0,
    fix="Run: python -m build",
)

tar_files = [f for f in os.listdir("dist") if f.endswith(".tar.gz")] \
    if os.path.isdir("dist") else []
check(
    f"Source dist (.tar.gz) exists: {tar_files[0] if tar_files else 'NOT FOUND'}",
    len(tar_files) > 0,
    fix="Run: python -m build",
)

check("pyproject.toml exists", os.path.exists("pyproject.toml"))
check("README.md exists",      os.path.exists("README.md"))
check("LICENSE exists",        os.path.exists("LICENSE"))

try:
    glob_pattern = os.path.join("dist", "*")
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", glob_pattern],
        capture_output=True, text=True, timeout=30,
    )
    passed_twine = "PASSED" in result.stdout or result.returncode == 0
    check("twine check passes (PyPI metadata valid)",
          passed_twine,
          fix=f"Output: {result.stdout.strip()}{result.stderr.strip()}")
except FileNotFoundError:
    check("twine installed", False, warn=True,
          fix="pip install twine")
except Exception as e:
    check("twine check", False, warn=True, fix=str(e))

# ── 8. GIT STATUS ─────────────────────────────────────────────────────
section("8. Git status")

try:
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, timeout=10,
    )
    uncommitted = r.stdout.strip()
    check(
        "No uncommitted changes",
        not uncommitted,
        fix=(
            f"git add . && git commit -m 'pre-launch cleanup'\n"
            f"         Uncommitted files:\n"
            + "\n".join(f"           {line}" for line in uncommitted.splitlines())
        ),
        warn=bool(uncommitted),
    )
    r2 = subprocess.run(
        ["git", "log", "--oneline", "-3"],
        capture_output=True, text=True, timeout=10,
    )
    log = r2.stdout.strip().replace("\n", " | ")
    check(f"Git commits present: {log[:70]}", bool(r2.stdout.strip()))
except FileNotFoundError:
    check("git installed", False, warn=True, fix="Install Git from https://git-scm.com")
except Exception as e:
    check("Git check", False, warn=True, fix=str(e))

# ── SUMMARY ───────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
print(
    f"{BOLD}RESULT: {GREEN}{passed} passed{RESET}{BOLD}  |  "
    f"{RED}{failed} failed{RESET}{BOLD}  |  "
    f"{YELLOW}{warn_count} warnings{RESET}"
)
print("=" * 55)

if failed == 0:
    print(f"\n{GREEN}{BOLD}ALL CHECKS PASSED — library is ready to publish!{RESET}")
    print("\nNext steps:")
    print("  1.  git push origin main")
    print("  2.  twine upload dist/*")
    print("  3.  pip install baxter-ts   (verify from PyPI)")
else:
    print(f"\n{RED}{BOLD}Fix the {failed} failed check(s) above before publishing.{RESET}")

print()
sys.exit(0 if failed == 0 else 1)
