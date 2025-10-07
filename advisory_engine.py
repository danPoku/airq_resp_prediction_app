import pandas as pd
from typing import List, Dict, Any, Tuple


def _last_n_days(df: pd.DataFrame, date_col: str, n: int) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.sort_values(date_col)
    # Use last n unique dates, not rows
    uniq = d.drop_duplicates([date_col], keep="last")
    tail_dates = uniq[date_col].dropna().unique()[-n:]
    return d[d[date_col].isin(tail_dates)].copy()


def _pct_change(start: float, end: float) -> float | None:
    if pd.isna(start) or start == 0:
        return None
    return (end - start) / start * 100.0


def _tier_from_change_and_hits(pct: float | None, high_hits: int) -> str:
    # Simple, interpretable rule for now
    if pct is None:
        pct_v = 0.0
    else:
        pct_v = abs(pct)
    if pct_v >= 20 or high_hits >= 3:
        return "high"
    if pct_v >= 10 or high_hits >= 1:
        return "medium"
    return "low"


def summarize_weekly_aq(df: pd.DataFrame, date_col: str, pollutant_cols: List[str]) -> Dict[str, Any]:
    """
    Summarize last ~7 days for air pollutants: change %, slope/day, mean7, high_days, tier.
    Uses data-driven high threshold (p75) from available history if possible.
    """
    if df is None or df.empty:
        return {"per_pollutant": {}, "top_concerns": []}

    d_all = df.copy()
    d_all[date_col] = pd.to_datetime(d_all[date_col]).dt.normalize()
    d_all = d_all.sort_values(date_col)
    d7 = _last_n_days(d_all, date_col, 7)

    # Thresholds: prefer p75 from all available dates when >= 14, else from last 7
    per_pollutant: Dict[str, Dict[str, Any]] = {}
    for col in pollutant_cols:
        series_all = pd.to_numeric(d_all[col], errors="coerce").dropna()
        series_7 = pd.to_numeric(d7[col], errors="coerce").dropna()
        if series_all.size >= 14:
            high_thr = float(series_all.quantile(0.75))
        elif series_7.size >= 3:
            high_thr = float(series_7.quantile(0.75))
        else:
            high_thr = float(series_all.quantile(0.75)) if series_all.size > 0 else float("nan")

        if series_7.empty:
            per_pollutant[col] = {
                "change_pct": None,
                "slope_per_day": None,
                "mean7": None,
                "high_days": 0,
                "tier": "low",
            }
            continue

        start = float(series_7.iloc[0])
        end = float(series_7.iloc[-1])
        pct = _pct_change(start, end)
        n = max(1, series_7.size - 1)
        slope = (end - start) / n
        mean7 = float(series_7.mean())
        high_days = int((series_7 > high_thr).sum()) if pd.notna(high_thr) else 0
        tier = _tier_from_change_and_hits(pct, high_days)

        per_pollutant[col] = {
            "change_pct": pct,
            "slope_per_day": slope,
            "mean7": mean7,
            "high_days": high_days,
            "tier": tier,
        }

    # Top concerns: prioritize high tier by magnitude of change
    ranked = sorted(
        (
            (col, vals)
            for col, vals in per_pollutant.items()
            if vals.get("change_pct") is not None
        ),
        key=lambda x: abs(x[1]["change_pct"]),
        reverse=True,
    )
    concerns = [
        f"{name}: {'+' if vals['change_pct']>=0 else ''}{vals['change_pct']:.1f}% over last week; {vals['high_days']} high-risk days"
        for name, vals in ranked
        if vals.get("tier") in {"high", "medium"}
    ][:3]

    return {"per_pollutant": per_pollutant, "top_concerns": concerns}


def precautions_for_tomorrow_aq(tomorrow: pd.Series, reference_df: pd.DataFrame, pollutant_cols: List[str]) -> List[str]:
    """Rule-based precautions for tomorrow. Uses p75 of reference_df as 'elevated'."""
    if tomorrow is None or tomorrow.empty:
        return []
    tips: List[str] = []
    ref = reference_df.copy()
    for col in pollutant_cols:
        if col not in tomorrow:
            continue
        thr = pd.to_numeric(ref[col], errors="coerce").quantile(0.75)
        val = pd.to_numeric(pd.Series([tomorrow[col]])).iloc[0]
        if pd.isna(val) or pd.isna(thr):
            continue
        if val >= thr:
            if col in ("pm2_5", "pm10"):
                tips.append("Limit outdoor exertion; close windows during peaks; use filtration if available.")
                tips.append("Sensitive groups (children, elderly, asthma/COPD) should wear masks outdoors.")
            elif col == "o3":
                tips.append("Avoid midday outdoor exercise; plan activities for early morning or evening.")
            elif col in ("no2", "co"):
                tips.append("Improve indoor ventilation; avoid idling vehicles and indoor solid-fuel burning.")
            elif col in ("so2", "nh3"):
                tips.append("Reduce exposure near industrial sources; use masks in affected areas.")
            # Coalesce duplicate lines later; keep simple now
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in tips:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:8]


def summarize_weekly_resp(df: pd.DataFrame, date_col: str, disease_cols: List[str]) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"per_disease": {}, "notes": []}
    d_all = df.copy()
    d_all[date_col] = pd.to_datetime(d_all[date_col]).dt.normalize()
    d_all = d_all.sort_values(date_col)
    d7 = _last_n_days(d_all, date_col, 7)

    per_disease: Dict[str, Dict[str, Any]] = {}
    for col in disease_cols:
        s7 = pd.to_numeric(d7[col], errors="coerce").dropna()
        if s7.empty:
            per_disease[col] = {"change_pct": None, "mean7": None, "load_tier": "low"}
            continue
        start = float(s7.iloc[0])
        end = float(s7.iloc[-1])
        pct = _pct_change(start, end)
        mean7 = float(s7.mean())
        # Simple tiering by relative change for now
        if pct is None:
            tier = "low"
        else:
            ap = abs(pct)
            tier = "high" if ap >= 30 else ("medium" if ap >= 15 else "low")
        per_disease[col] = {"change_pct": pct, "mean7": mean7, "load_tier": tier}

    notes = [
        f"{name}: {'+' if v['change_pct']>=0 else ''}{v['change_pct']:.1f}% over last week"
        for name, v in per_disease.items() if v.get("change_pct") is not None
    ]
    return {"per_disease": per_disease, "notes": notes}


def hospital_advisory_for_tomorrow(
    df: pd.DataFrame,
    date_col: str,
    disease_cols: List[str],
    tomorrow_ts: pd.Timestamp,
) -> Dict[str, Any]:
    """Inventory/ops checklist for expected respiratory cases tomorrow."""
    if df is None or df.empty:
        return {"by_disease": {}, "summary": []}

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
    d = d.sort_values(date_col)
    if tomorrow_ts not in set(d[date_col]):
        # pick nearest forward date if exact not present
        tomorrow_ts = d[date_col].max()

    # previous 7 days for baseline
    d_prev = d[d[date_col] < tomorrow_ts].tail(7)

    by_disease: Dict[str, Dict[str, Any]] = {}

    inventory_map: Dict[str, List[str]] = {
        "Asthma (J45)": [
            "Short-acting bronchodilators (SABA)",
            "Inhaled corticosteroids", "Spacers", "Nebulizers", "Peak-flow meters",
            "Pulse oximeters", "Oxygen supply check",
        ],
        "Pneumonia (J12-J18)": [
            "Oxygen cylinders/concentrators", "Suction kits", "IV cannulas/fluids",
            "Antibiotics per protocol", "PPE", "Rapid diagnostics (if available)",
        ],
        "Upper Respiratory Tract Infection (J00-J06)": [
            "Analgesics/antipyretics", "Rapid tests (as applicable)", "Masks/PPE",
        ],
        "Chronic Obstructive Pulmonary Disease (J44)": [
            "Oxygen delivery sets", "NIV masks & circuits", "Corticosteroids",
        ],
        "Bronchiolitis (J21)": [
            "Pediatric nasal cannulas/masks", "Humidifiers", "Suction bulbs",
        ],
        "Influenza (J09-J11)": [
            "Antivirals (as per guidance)", "PPE", "Rapid influenza tests",
        ],
        "Acute Bronchitis (J20)": [
            "Cough suppressants/expectorants", "Supportive care supplies",
        ],
    }

    ops_common = [
        "Review triage staffing for peak hours",
        "Check oxygen plant/concentrator capacity",
        "Prepare surge beds or fast-track pathway if needed",
    ]

    summary: List[str] = []
    row_t = d[d[date_col] == tomorrow_ts].tail(1)
    for col in disease_cols:
        expected = int(pd.to_numeric(row_t[col], errors="coerce").fillna(0).iloc[0]) if not row_t.empty else 0
        baseline = float(pd.to_numeric(d_prev[col], errors="coerce").mean()) if not d_prev.empty else 0.0
        pct = _pct_change(baseline, expected) if baseline not in (0, None) else None
        tier = "high" if (pct is not None and abs(pct) >= 30) or expected >= max(10, baseline * 1.3) else (
            "medium" if (pct is not None and abs(pct) >= 15) or expected >= max(5, baseline * 1.15) else "low"
        )
        by_disease[col] = {
            "expected": expected,
            "baseline7": baseline,
            "delta_pct": pct,
            "load_tier": tier,
            "inventory": inventory_map.get(col, []),
            "ops": ops_common,
        }
        if tier in ("medium", "high"):
            delta_txt = f"{('+' if (pct or 0)>=0 else '')}{pct:.1f}%" if pct is not None else "N/A"
            summary.append(f"{col}: {expected} expected (vs ~{baseline:.0f}); {delta_txt}, tier {tier}")

    return {"by_disease": by_disease, "summary": summary[:6]}


# Optional: LLM integration scaffolding (disabled by default)
def compose_llm_prompt(aq_summary: Dict[str, Any] | None, hosp_advice: Dict[str, Any] | None) -> str:
    lines: List[str] = [
        "You are an air quality and public health operations assistant.",
        "Write a concise advisory (<= 200 words) in bullet points.",
    ]
    if aq_summary:
        lines.append("Air Quality Weekly Summary:")
        lines.extend([f"- {t}" for t in aq_summary.get("top_concerns", [])])
    if hosp_advice:
        lines.append("Hospital Tomorrow Summary:")
        lines.extend([f"- {t}" for t in hosp_advice.get("summary", [])])
    lines.append("Conclude with 2 key precautions.")
    return "\n".join(lines)


def generate_llm_advisory(prompt: str) -> str | None:
    """
    Placeholder for LLM call. Hook up your provider here.
    Return None if not configured; the app will fall back to rule-based text.
    """
    try:
        # Example (pseudo):
        # import openai
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        # resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}])
        # return resp.choices[0].message.content
        return None
    except Exception:
        return None

