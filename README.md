# 🌱 Green Bond Pricing & ESG Analytics Engine

## Overview

This project is a **professional-grade, end-to-end Green Bond analytics engine** designed to replicate how **asset managers, banks, and institutional investors** price green bonds while simultaneously assessing **ESG quality and regulatory classification**.

The tool combines:

* **Fixed income pricing (buy-side standards)**
* **Quantitative ESG scoring**
* **Green Bond regulatory frameworks (ICMA, EU Taxonomy, SFDR)**
* **A unified KPI dashboard**, suitable for investment committees or ESG reporting

All logic is implemented in **a single deterministic Python file**, making it ideal for **research, interviews, GitHub portfolios, or internal prototypes**.

---

## Key Features

### 1️ Fixed Income Pricing Engine

The model prices a standard fixed-rate green bond using a **risk-free zero-coupon curve plus credit spread**.

**Computed Financial KPIs:**

* Clean price
* Dirty price
* Yield to Maturity (YTM)
* Macaulay duration
* Modified duration
* Convexity
* DV01
* Z-spread
* Credit spread (input)
* **Greenium** (vs. comparable vanilla bond)

Methodology aligns with **buy-side fixed income analytics** (discounted cash flows, continuous compounding for curves, YTM-based risk metrics).

---

### 2️ Quantitative ESG Scoring Model

A fully transparent **ESG scoring framework** with explicit normalization, weights, and rating logic.

#### Environmental (40%)

* CO₂ avoided (tons/year)
* Energy efficiency improvement (%)
* Renewable energy share (%)
* Water usage reduction (%)
* Climate alignment score (Paris-aligned proxy)

#### Social (30%)

* Jobs created
* Community impact
* Health & safety indicators
* Inclusion / access to essential services

#### Governance (30%)

* Board independence (%)
* ESG transparency score
* Reporting frequency
* External audit / verification

**Outputs:**

* E / S / G sub-scores (0–100)
* Final ESG score (0–100)
* ESG rating (**AAA → CCC**, MSCI-inspired)
* Climate impact indicators

All assumptions are **deterministic and auditable** (no black box).

---

### 3️ Green Bond Regulatory Classification

The engine classifies the bond according to **current European and international standards**:

* **ICMA Green Bond Principles** (scored & compliant / non-compliant)
* **EU Taxonomy alignment (%)**
* **SFDR classification**: Article 6 / 8 / 9

This reflects how green bonds are assessed in **asset management, insurance, and banking regulation**.

---

### 4️ Unified KPI Dashboard

All results are aggregated into a **structured Python dictionary**:

```python
{
  "financial_kpis": {...},
  "esg_kpis": {...},
  "regulatory_kpis": {...}
}
```

This format is ideal for:

* Investment committee notes
* ESG reporting
* Risk dashboards
* Further integration (API, Streamlit, Excel, BI tools)

---

## Project Structure

```
├── green_bond.py   # Single-file end-to-end analytics engine
└── README.md
```

No external data sources or APIs are required.

---

## Example Use Case

The script includes a **deterministic example**:

* 5Y semi-annual green bond
* Stylized EUR risk-free curve
* Comparable vanilla bond (to compute greenium)
* Realistic ESG and regulatory inputs

Running the file directly:

```bash
python green_bond.py
```

Outputs a **fully populated KPI dashboard** printed to the console.

---

## Intended Audience

This project is suitable for:

* Fixed Income Analysts
* ESG Analysts
* Asset Managers
* Risk & Quant teams
* Finance students targeting **buy-side / sell-side roles**

It is intentionally designed to resemble an **internal institutional research tool**, not a toy model.

---

## Methodological Notes

* Yield and duration metrics are computed under **standard market conventions**
* ESG weights and thresholds are **explicit and easily adjustable**
* Regulatory logic is simplified but **conceptually aligned with EU frameworks**

---

## Disclaimer

This project is for **educational and research purposes only**.
It does not constitute investment advice and should not be used for live trading without validation against market data.

---

## Author

**Toussaint Yonga**
Finance | Fixed Income | ESG | Quantitative Research

---

If you are a recruiter, portfolio manager, or ESG specialist, this repository is intended to demonstrate:

* Financial engineering rigor
* ESG analytical depth
* Regulatory awareness
* Clean, auditable Python design
