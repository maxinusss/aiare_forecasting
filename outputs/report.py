"""
Generate a self-contained HTML report summarising the opus_forecast ensemble
outputs for AIARE course enrollment forecasting.

Reads artefacts from  forecast/opus_analysis/  and writes
outputs/aiare_forecast_report.html.

Usage:
    python outputs/report.py
"""

import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "forecast" / "opus_analysis_updated2526_lower_overforecast_penalty"
OUTPUT_HTML = ROOT / "outputs" / "aiare_forecast_report_1.5xpenalty.html"

# 26/27 winter season boundaries
ANNUAL_START = pd.Timestamp("2026-07-01")
ANNUAL_END = pd.Timestamp("2027-07-01")  # exclusive


# ── helpers ────────────────────────────────────────────────────────────────
def img_to_base64(path: Path) -> str:
    """Read a PNG and return a base64-encoded data-URI string."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def fmt(val, decimals=1):
    """Format a number nicely; return '–' for NaN."""
    if pd.isna(val):
        return "–"
    return f"{val:,.{decimals}f}"


def sanitize_name(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def season_label(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Spring"
    if m in [6, 7, 8]:
        return "Summer"
    return "Fall"


# ── load data ──────────────────────────────────────────────────────────────
def load_all():
    forecasts = pd.read_csv(ANALYSIS_DIR / "all_courses_future_forecasts.csv")
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    leaderboard = pd.read_csv(ANALYSIS_DIR / "course_model_leaderboard.csv")

    course_dirs = sorted(
        [d for d in ANALYSIS_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    course_data = {}
    for cdir in course_dirs:
        name = cdir.name  # e.g. "aiare_1"

        summary_path = cdir / "best_model_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        forecast_path = cdir / "future_forecast.csv"
        forecast = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
        if not forecast.empty:
            forecast["date"] = pd.to_datetime(forecast["date"])

        history_img = img_to_base64(cdir / "history_and_forecast.png")

        course_data[name] = {
            "summary": summary,
            "forecast": forecast,
            "history_img": history_img,
        }


    combined_img = img_to_base64(ANALYSIS_DIR / "combined_course_forecasts.png")

    return forecasts, leaderboard, course_data, combined_img


# ── annual summary ─────────────────────────────────────────────────────────
def compute_annual_summary(forecasts):
    """Sum forecasts within the 26/27 winter season (Jul 2026 – Jun 2027)."""
    mask = (forecasts["date"] >= ANNUAL_START) & (forecasts["date"] < ANNUAL_END)
    annual = forecasts.loc[mask].copy()
    annual["season"] = annual["date"].apply(season_label)

    by_course = (
        annual.groupby("combined_course")
        .agg(
            total_students=("forecast_num_students", "sum"),
            peak_month_students=("forecast_num_students", "max"),
        )
        .reset_index()
    )
    by_course["total_students"] = by_course["total_students"].round(0).astype(int)
    by_course["peak_month_students"] = by_course["peak_month_students"].round(0).astype(int)

    grand_total = int(by_course["total_students"].sum())

    by_course_season = (
        annual.groupby(["combined_course", "season"])["forecast_num_students"]
        .sum()
        .round(0)
        .astype(int)
        .unstack(fill_value=0)
    )
    # ensure season order
    for s in ["Summer", "Fall", "Winter", "Spring"]:
        if s not in by_course_season.columns:
            by_course_season[s] = 0
    by_course_season = by_course_season[["Summer", "Fall", "Winter", "Spring"]]

    return by_course, grand_total, by_course_season, annual


# ── HTML generation ────────────────────────────────────────────────────────
def build_html(forecasts, leaderboard, course_data, combined_img):
    annual_by_course, grand_total, seasonal_pivot, annual_detail = compute_annual_summary(forecasts)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    course_names = sorted(course_data.keys())

    # ---- start HTML ----
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIARE Enrollment Forecast Report – 2026/27 Season</title>
<style>
  :root {{
    --bg: #f8f9fa; --card: #fff; --accent: #2563eb; --text: #1e293b;
    --muted: #64748b; --border: #e2e8f0; --success: #16a34a; --warn: #d97706;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6;
         padding: 2rem; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.35rem; color: var(--accent); margin: 2rem 0 0.75rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; }}
  h3 {{ font-size: 1.1rem; margin: 1.25rem 0 0.5rem; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1.25rem; margin-bottom: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .kpi-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }}
  .kpi {{ flex: 1; min-width: 180px; background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 1rem; text-align: center; }}
  .kpi .value {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
  .kpi .label {{ font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin-top: 0.5rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: right; border-bottom: 1px solid var(--border); }}
  th {{ background: #f1f5f9; font-weight: 600; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  tr:hover {{ background: #f8fafc; }}
  .img-container {{ text-align: center; margin: 1rem 0; }}
  .img-container img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); }}
  .tag {{ display: inline-block; background: #e0e7ff; color: #3730a3; padding: 0.15rem 0.5rem;
          border-radius: 4px; font-size: 0.78rem; font-weight: 500; margin-right: 0.3rem; }}
  .notes {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }}
  .notes li {{ margin-bottom: 0.2rem; }}
  .ensemble-detail {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .ensemble-detail > div {{ flex: 1; min-width: 280px; }}
  .footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
</style>
</head>
<body>

<h1>AIARE Course Enrollment Forecast Report</h1>
<p class="subtitle">2026/27 Winter Season (Jul 2026 – Jun 2027) &bull; Generated {generated}</p>
""")

    # ── Executive Summary ──────────────────────────────────────────────
    parts.append('<h2>Executive Summary</h2>')
    parts.append('<div class="kpi-row">')
    parts.append(f'<div class="kpi"><div class="value">{grand_total:,}</div><div class="label">Total Forecast Students<br>(Jul \'26 – Jun \'27)</div></div>')
    for _, row in annual_by_course.iterrows():
        parts.append(
            f'<div class="kpi"><div class="value">{row["total_students"]:,}</div>'
            f'<div class="label">{row["combined_course"].title()}<br>Annual Total</div></div>'
        )
    parts.append('</div>')

    # seasonal breakdown
    parts.append('<div class="card">')
    parts.append('<h3>Seasonal Breakdown (Jul 2026 – Jun 2027)</h3>')
    parts.append('<table><thead><tr><th>Course</th><th>Summer<br>(Jul-Aug)</th><th>Fall<br>(Sep-Nov)</th><th>Winter<br>(Dec-Feb)</th><th>Spring<br>(Mar-Jun)</th><th>Annual Total</th></tr></thead><tbody>')
    for course in seasonal_pivot.index:
        total = int(seasonal_pivot.loc[course].sum())
        parts.append(
            f'<tr><td>{course}</td>'
            f'<td>{seasonal_pivot.loc[course, "Summer"]:,}</td>'
            f'<td>{seasonal_pivot.loc[course, "Fall"]:,}</td>'
            f'<td>{seasonal_pivot.loc[course, "Winter"]:,}</td>'
            f'<td>{seasonal_pivot.loc[course, "Spring"]:,}</td>'
            f'<td><strong>{total:,}</strong></td></tr>'
        )
    # totals row
    parts.append(
        f'<tr style="font-weight:700;border-top:2px solid var(--accent)"><td>All Courses</td>'
        f'<td>{seasonal_pivot["Summer"].sum():,}</td>'
        f'<td>{seasonal_pivot["Fall"].sum():,}</td>'
        f'<td>{seasonal_pivot["Winter"].sum():,}</td>'
        f'<td>{seasonal_pivot["Spring"].sum():,}</td>'
        f'<td>{grand_total:,}</td></tr>'
    )
    parts.append('</tbody></table></div>')

    # ── Combined forecast chart ────────────────────────────────────────
    if combined_img:
        parts.append('<div class="card">')
        parts.append('<h3>Combined Course Forecasts</h3>')
        parts.append(f'<div class="img-container"><img src="{combined_img}" alt="Combined forecast chart"></div>')
        parts.append('</div>')

    # ── Model Leaderboard ──────────────────────────────────────────────
    parts.append('<h2>Model Leaderboard</h2>')
    parts.append('<div class="card">')
    parts.append('<table><thead><tr><th>Course</th><th>Best Model</th><th>CV RMSE</th><th>CV MAE</th><th>CV MAPE (%)</th><th>Ensemble Size</th></tr></thead><tbody>')
    for _, row in leaderboard.iterrows():
        parts.append(
            f'<tr><td>{row["course"]}</td>'
            f'<td><span class="tag">{row["best_model_family"]}</span></td>'
            f'<td>{fmt(row["best_cv_rmse"])}</td>'
            f'<td>{fmt(row["best_cv_mae"])}</td>'
            f'<td>{fmt(row["best_cv_mape"])}</td>'
            f'<td>{row["ensemble_n_models"]}</td></tr>'
        )
    parts.append('</tbody></table></div>')

    # ── Per-course detail sections ─────────────────────────────────────
    for cname in course_names:
        cd = course_data[cname]
        summary = cd["summary"]
        course_title = summary.get("course", cname).title()

        parts.append(f'<h2>{course_title}</h2>')

        # ensemble info
        parts.append('<div class="card"><div class="ensemble-detail">')

        parts.append('<div>')
        parts.append(f'<h3>Best Model: <span class="tag">{summary.get("best_model_family", "–")}</span></h3>')
        params = summary.get("best_model_params", {})
        if params:
            parts.append(f'<p style="font-size:0.85rem;color:var(--muted)">Params: {json.dumps(params)}</p>')
        parts.append(f'<p>Train period: {summary.get("train_start","?")} → {summary.get("train_end","?")}</p>')
        parts.append(f'<p>Forecast period: {summary.get("forecast_start","?")} → {summary.get("forecast_end","?")}</p>')
        parts.append(f'<p>Over-forecast penalty: {summary.get("over_forecast_penalty", "–")}×</p>')
        parts.append('</div>')

        parts.append('<div>')
        parts.append(f'<h3>Ensemble ({summary.get("ensemble_n_models", "–")} models)</h3>')
        weights = summary.get("ensemble_weights", {})
        families = summary.get("ensemble_families", [])
        if weights:
            parts.append('<table><thead><tr><th>Model</th><th>Weight</th></tr></thead><tbody>')
            for fam in families:
                w = weights.get(fam, "–")
                w_str = f"{w:.1%}" if isinstance(w, (int, float)) else str(w)
                parts.append(f'<tr><td>{fam}</td><td>{w_str}</td></tr>')
            parts.append('</tbody></table>')
        parts.append('</div>')

        parts.append('<div>')
        parts.append('<h3>CV Metrics</h3>')
        parts.append('<table>')
        parts.append(f'<tr><td>RMSE</td><td>{fmt(summary.get("best_cv_rmse"))}</td></tr>')
        parts.append(f'<tr><td>MAE</td><td>{fmt(summary.get("best_cv_mae"))}</td></tr>')
        parts.append(f'<tr><td>MAPE</td><td>{fmt(summary.get("best_cv_mape"))}%</td></tr>')
        parts.append(f'<tr><td>Asymmetric Loss</td><td>{fmt(summary.get("best_cv_asym_loss"))}</td></tr>')
        parts.append('</table>')
        parts.append('</div>')

        parts.append('</div></div>')  # end ensemble-detail, card

        # history + forecast chart
        if cd["history_img"]:
            parts.append('<div class="card">')
            parts.append('<h3>Historical Enrollment &amp; Forecast</h3>')
            parts.append(f'<div class="img-container"><img src="{cd["history_img"]}" alt="{course_title} forecast chart"></div>')
            parts.append('</div>')

        # monthly forecast table (annual period)
        fc = cd["forecast"]
        if not fc.empty:
            fc_annual = fc[(fc["date"] >= ANNUAL_START) & (fc["date"] < ANNUAL_END)].copy()
            if not fc_annual.empty:
                parts.append('<div class="card">')
                parts.append('<h3>Monthly Forecast – 26/27 Season (Jul 2026 – Jun 2027)</h3>')
                parts.append('<table><thead><tr><th>Month</th><th>Ensemble Forecast</th><th>Best Model Only</th><th>Season</th></tr></thead><tbody>')
                season_total_ensemble = 0
                season_total_best = 0
                for _, r in fc_annual.iterrows():
                    ens = r["forecast_num_students"]
                    best = r["forecast_single_best"]
                    sl = season_label(r["date"])
                    season_total_ensemble += ens
                    season_total_best += best
                    parts.append(
                        f'<tr><td>{r["date"].strftime("%b %Y")}</td>'
                        f'<td>{fmt(ens, 0)}</td>'
                        f'<td>{fmt(best, 0)}</td>'
                        f'<td>{sl}</td></tr>'
                    )
                parts.append(
                    f'<tr style="font-weight:700;border-top:2px solid var(--accent)">'
                    f'<td>Total</td><td>{fmt(season_total_ensemble, 0)}</td>'
                    f'<td>{fmt(season_total_best, 0)}</td><td></td></tr>'
                )
                parts.append('</tbody></table></div>')

        # methodology notes
        notes = summary.get("notes", [])
        if notes:
            parts.append('<div class="card">')
            parts.append('<h3>Methodology Notes</h3>')
            parts.append('<ul class="notes">')
            for n in notes:
                parts.append(f'<li>{n}</li>')
            parts.append('</ul></div>')

    # ── Full monthly forecast table (all courses, all dates) ───────────
    parts.append('<h2>Full Monthly Forecast – All Courses</h2>')
    parts.append('<div class="card">')
    pivot = forecasts.pivot_table(
        index="date", columns="combined_course",
        values="forecast_num_students", aggfunc="sum",
    ).fillna(0)
    pivot["Total"] = pivot.sum(axis=1)
    parts.append('<table><thead><tr><th>Month</th>')
    for col in pivot.columns:
        parts.append(f'<th>{col.title() if col != "Total" else "<strong>Total</strong>"}</th>')
    parts.append('</tr></thead><tbody>')
    for dt, row in pivot.iterrows():
        hl = ' style="background:#eff6ff"' if ANNUAL_START <= dt < ANNUAL_END else ""
        parts.append(f'<tr{hl}><td>{dt.strftime("%b %Y")}</td>')
        for col in pivot.columns:
            parts.append(f'<td>{fmt(row[col], 0)}</td>')
        parts.append('</tr>')
    # grand totals
    parts.append('<tr style="font-weight:700;border-top:2px solid var(--accent)"><td>Grand Total</td>')
    for col in pivot.columns:
        parts.append(f'<td>{fmt(pivot[col].sum(), 0)}</td>')
    parts.append('</tr></tbody></table></div>')

    # ── Footer ─────────────────────────────────────────────────────────
    parts.append(f"""
<div class="footer">
  AIARE Enrollment Forecasting &bull; Opus Forecast Pipeline &bull; Report generated {generated}<br>
  26/27 season rows highlighted in blue in full table above.
</div>
</body></html>""")

    return "\n".join(parts)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    forecasts, leaderboard, course_data, combined_img = load_all()
    html = build_html(forecasts, leaderboard, course_data, combined_img)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report written to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
