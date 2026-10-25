# Customer Churn / Sales Trend Analysis

End-to-end analysis of an e-commerce order table to find revenue trends, the best product categories, the most valuable customers and customers at risk of churning, finished with a one-page business brief.

Built for **InternCircle - Data Science & AI - Project 03**.

> **Data note:** the data is **simulated** (fixed random seed 42, so results are reproducible). No real customer data is used. The simulation includes a festive-season (Oct-Dec) lift, different order values per category and uneven customer activity, so the findings describe this made-up dataset, not a real company.

## Headline results (simulated data)

| Question | Answer |
|---|---|
| Total revenue / avg order | INR 2.98M / INR 595 |
| Best / worst month | Nov 2025 (INR 323k) / Feb 2025 (INR 187k), 1.7x apart |
| Q4 share of revenue | 30.9% |
| Top category | Electronics, 45.7% of revenue |
| Top 10% of customers | 36.2% of revenue (not a strict 80/20) |
| Customers inactive 90+ days | 35 (8.8%), none in the top 10% |

## Project structure

```
.
├── data/orders.csv                  # generated order table (5,000 rows)
├── src/analysis.py                  # full pipeline: data, features, trends, charts, metrics
├── notebooks/sales_trend_analysis.ipynb   # step-by-step notebook with outputs
├── outputs/
│   ├── charts/                      # 4 PNG charts (one message per chart)
│   ├── monthly_trend.csv            # revenue + month-over-month growth
│   ├── category_summary.csv         # total, average, count per category
│   ├── month_by_category_pivot.csv  # pivot table: month x category
│   ├── customer_summary.csv         # spend, orders, recency, segment per customer
│   └── metrics.json                 # every number quoted in the brief
├── reports/
│   ├── business_brief.md            # one-page brief (Context, Findings, Recommendations, Next steps)
│   └── business_brief.pdf
└── requirements.txt
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/analysis.py             # regenerates data, tables, charts, metrics.json
jupyter notebook notebooks/sales_trend_analysis.ipynb
```

The notebook also runs on Google Colab: upload it and run all cells (it generates its own data).

## Method

1. Business question written first.
2. Data generated, then checked (shape, types, missing values, duplicates).
3. Date features: month, weekday, quarter.
4. Monthly revenue, month-over-month growth with `pct_change()`, best and worst month marked.
5. Category comparison with `groupby` and a month-by-category `pivot_table`.
6. Pareto test: revenue share of the top 10% / 20% of customers, plus customer segments.
7. Churn view: customers with no order in the last 90 days.
8. Three number-backed recommendations in `reports/business_brief.md`.
