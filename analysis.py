"""Customer Churn / Sales Trend Analysis - InternCircle Data Science & AI, Project 03.

Run from the project root:
    python src/analysis.py

Steps follow the learner guide:
 1. business question  2. create data + checks  3. date features
 4. monthly trend      5. categories + pivot    6. top customers (Pareto)
 + churn view (recency) -> metrics.json, charts, CSV tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
for d in (DATA_DIR, OUT_DIR, CHART_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_ORDERS = 5000
N_CUSTOMERS = 400
CATEGORIES = ["Electronics", "Fashion", "Home", "Books"]
CAT_PROBS = [0.30, 0.35, 0.20, 0.15]
# Average order value differs by category (simulated assumption).
CAT_SCALE = {"Electronics": 900, "Fashion": 450, "Home": 600, "Books": 250}
# Mild seasonality: festive-season lift in Oct-Dec (simulated assumption).
MONTH_WEIGHT = np.array([0.9, 0.85, 0.9, 0.95, 1.0, 1.0, 0.95, 1.0, 1.1, 1.25, 1.4, 1.35])

PRIMARY = "#1F4E79"
ACCENT = "#C8102E"
MUTED = "#8A94A6"


def make_orders() -> pd.DataFrame:
    """Create a made-up e-commerce order table (no real customer data)."""
    rng = np.random.default_rng(SEED)
    # Customers are not equally active: a skewed weight creates a realistic 80/20 pattern.
    cust_weight = rng.pareto(1.6, N_CUSTOMERS) + 1
    cust_weight /= cust_weight.sum()
    customer_id = rng.choice(np.arange(1, N_CUSTOMERS + 1), N_ORDERS, p=cust_weight)

    # Dates across 2025, weighted by month for seasonality.
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    day_w = MONTH_WEIGHT[days.month - 1]
    day_w = day_w / day_w.sum()
    order_date = rng.choice(days, N_ORDERS, p=day_w)

    category = rng.choice(CATEGORIES, N_ORDERS, p=CAT_PROBS)
    scale = np.array([CAT_SCALE[c] for c in category])
    amount = (rng.gamma(2.0, 1.0, N_ORDERS) * scale / 2).round(2)

    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, N_ORDERS + 1),
            "order_date": order_date,
            "customer_id": customer_id,
            "category": category,
            "amount": amount,
        }
    ).sort_values("order_date", ignore_index=True)
    orders["order_id"] = np.arange(1, len(orders) + 1)
    return orders


def data_checks(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
        "date_min": str(df["order_date"].min().date()),
        "date_max": str(df["order_date"].max().date()),
        "customers": int(df["customer_id"].nunique()),
    }


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    # ---- 2. Create data and check it
    orders = make_orders()
    checks = data_checks(orders)
    orders.to_csv(DATA_DIR / "orders.csv", index=False)

    # ---- 3. Date features
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["month"] = orders["order_date"].dt.to_period("M")
    orders["month_name"] = orders["order_date"].dt.strftime("%b")
    orders["weekday"] = orders["order_date"].dt.day_name()
    orders["quarter"] = "Q" + orders["order_date"].dt.quarter.astype(str)

    total_rev = orders["amount"].sum()

    # ---- 4. Monthly trend
    monthly = orders.groupby("month")["amount"].sum()
    growth = (monthly.pct_change() * 100).round(1)
    monthly_df = pd.DataFrame({"revenue": monthly.round(0), "mom_growth_pct": growth})
    monthly_df.index = monthly_df.index.astype(str)
    monthly_df.to_csv(OUT_DIR / "monthly_trend.csv")
    best_m, worst_m = monthly.idxmax(), monthly.idxmin()

    quarterly = orders.groupby("quarter")["amount"].sum()
    weekday_rev = (
        orders.groupby("weekday")["amount"]
        .sum()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = monthly.index.astype(str)
    ax.plot(x, monthly.values, marker="o", color=PRIMARY, lw=2)
    ax.scatter([str(best_m)], [monthly.max()], color="#2E8B57", s=120, zorder=5, label=f"Best: {best_m}")
    ax.scatter([str(worst_m)], [monthly.min()], color=ACCENT, s=120, zorder=5, label=f"Worst: {worst_m}")
    ax.set_title("Monthly revenue peaks in the festive season", loc="left", fontweight="bold")
    ax.set_ylabel("Revenue (INR)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e3:,.0f}k")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "01_monthly_revenue.png", dpi=150)
    plt.close(fig)

    # ---- 5. Categories
    by_cat = (
        orders.groupby("category")["amount"]
        .agg(total="sum", average="mean", orders="count")
        .sort_values("total", ascending=False)
    )
    by_cat["share_pct"] = (by_cat["total"] / total_rev * 100).round(1)
    by_cat.round(1).to_csv(OUT_DIR / "category_summary.csv")

    pivot = orders.pivot_table(
        index="month", columns="category", values="amount", aggfunc="sum", fill_value=0
    )[by_cat.index.tolist()]
    pivot_out = pivot.round(0)
    pivot_out.index = pivot_out.index.astype(str)
    pivot_out.to_csv(OUT_DIR / "month_by_category_pivot.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(by_cat.index, by_cat["total"], color=PRIMARY)
    for b, share in zip(bars, by_cat["share_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{share:.0f}%", ha="center", va="bottom")
    ax.set_title("Electronics and Fashion drive most revenue", loc="left", fontweight="bold")
    ax.set_ylabel("Revenue (INR)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v/1e6:,.1f}M")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "02_category_revenue.png", dpi=150)
    plt.close(fig)

    # ---- 6. Best customers (Pareto / 80-20)
    spend = orders.groupby("customer_id")["amount"].sum().sort_values(ascending=False)
    n_cust = len(spend)
    top10_n = max(n_cust // 10, 1)
    top10_share = spend.head(top10_n).sum() / spend.sum()
    top20_share = spend.head(max(n_cust // 5, 1)).sum() / spend.sum()
    cum = spend.cumsum() / spend.sum()
    n_for_80 = int((cum < 0.8).sum() + 1)

    cust = (
        orders.groupby("customer_id")
        .agg(
            total_spend=("amount", "sum"),
            orders=("order_id", "count"),
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
        )
        .sort_values("total_spend", ascending=False)
    )
    cust["avg_order_value"] = cust["total_spend"] / cust["orders"]
    cust["segment"] = np.where(
        np.arange(len(cust)) < top10_n,
        "Top 10%",
        np.where(np.arange(len(cust)) < n_cust * 0.5, "Middle 40%", "Bottom 50%"),
    )
    seg = cust.groupby("segment").agg(
        customers=("total_spend", "size"),
        revenue=("total_spend", "sum"),
        avg_orders=("orders", "mean"),
    )
    seg["revenue_share_pct"] = (seg["revenue"] / total_rev * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pct_customers = np.arange(1, n_cust + 1) / n_cust * 100
    ax.plot(pct_customers, cum.values * 100, color=PRIMARY, lw=2)
    ax.axvline(10, color=MUTED, ls="--", lw=1)
    ax.scatter([10], [top10_share * 100], color=ACCENT, s=90, zorder=5)
    ax.annotate(
        f"Top 10% of customers\n= {top10_share:.0%} of revenue",
        xy=(10, top10_share * 100),
        xytext=(25, top10_share * 100 - 18),
        arrowprops=dict(arrowstyle="->", color=MUTED),
    )
    ax.set_xlabel("% of customers (ranked by spend)")
    ax.set_ylabel("Cumulative % of revenue")
    ax.set_title("A small group of customers brings much of the revenue", loc="left", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "03_pareto_customers.png", dpi=150)
    plt.close(fig)

    # ---- Churn view: who has gone quiet? (recency)
    snapshot = orders["order_date"].max() + pd.Timedelta(days=1)
    cust["days_since_last"] = (snapshot - cust["last_order"]).dt.days
    cust["churn_risk"] = cust["days_since_last"] > 90
    churned = cust["churn_risk"].sum()
    churn_rate = churned / n_cust
    top_at_risk = cust[(cust["segment"] == "Top 10%") & cust["churn_risk"]]
    rev_at_risk_top = top_at_risk["total_spend"].sum()
    # Repeat vs one-time buyers
    one_time = int((cust["orders"] == 1).sum())
    num_cols = cust.select_dtypes("number").columns
    cust_out = cust.copy()
    cust_out[num_cols] = cust_out[num_cols].round(2)
    cust_out.to_csv(OUT_DIR / "customer_summary.csv")
    risk_by_seg = cust.groupby("segment").agg(
        customers=("churn_risk", "size"), at_risk=("churn_risk", "sum"), at_risk_spend=("total_spend", lambda x: x[cust.loc[x.index, "churn_risk"]].sum())
    )
    risk_by_seg["at_risk_pct"] = (risk_by_seg["at_risk"] / risk_by_seg["customers"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cust["days_since_last"], bins=25, color=PRIMARY)
    ax.axvline(90, color=ACCENT, ls="--")
    ax.text(92, ax.get_ylim()[1] * 0.9, f"90-day cut-off\n{churn_rate:.0%} of customers beyond it", color=ACCENT)
    ax.set_xlabel("Days since last order")
    ax.set_ylabel("Customers")
    ax.set_title("Most customers ordered recently; a tail has gone quiet", loc="left", fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "04_recency_churn.png", dpi=150)
    plt.close(fig)

    # ---- Metrics for the brief
    top_cat = by_cat.index[0]
    q4_share = quarterly.get("Q4", 0) / total_rev
    metrics = {
        "checks": checks,
        "total_revenue": round(float(total_rev), 0),
        "avg_order_value": round(float(orders["amount"].mean()), 1),
        "best_month": str(best_m),
        "best_month_revenue": round(float(monthly.max()), 0),
        "worst_month": str(worst_m),
        "worst_month_revenue": round(float(monthly.min()), 0),
        "best_vs_worst_ratio": round(float(monthly.max() / monthly.min()), 2),
        "biggest_mom_jump": {"month": str(growth.idxmax()), "pct": float(growth.max())},
        "biggest_mom_drop": {"month": str(growth.idxmin()), "pct": float(growth.min())},
        "q4_share_pct": round(float(q4_share * 100), 1),
        "quarterly_revenue": {k: round(float(v), 0) for k, v in quarterly.items()},
        "weekday_best": str(weekday_rev.idxmax()),
        "weekday_worst": str(weekday_rev.idxmin()),
        "categories": by_cat.round(1).reset_index().to_dict(orient="records"),
        "top_category": top_cat,
        "top10_share_pct": round(float(top10_share * 100), 1),
        "top20_share_pct": round(float(top20_share * 100), 1),
        "customers_for_80pct_revenue": n_for_80,
        "customers_for_80pct_share_of_base_pct": round(n_for_80 / n_cust * 100, 1),
        "top10_customer_count": top10_n,
        "segments": seg.round(1).reset_index().to_dict(orient="records"),
        "churn_cutoff_days": 90,
        "churn_risk_customers": int(churned),
        "churn_risk_rate_pct": round(float(churn_rate * 100), 1),
        "top10_at_risk_customers": int(len(top_at_risk)),
        "top10_at_risk_lifetime_spend": round(float(rev_at_risk_top), 0),
        "churn_risk_by_segment": risk_by_seg.round(1).reset_index().to_dict(orient="records"),
        "one_time_buyers": one_time,
        "one_time_buyers_pct": round(one_time / n_cust * 100, 1),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nMonthly table:\n", monthly_df)
    print("\nPivot head:\n", pivot_out.head(3))


if __name__ == "__main__":
    main()
