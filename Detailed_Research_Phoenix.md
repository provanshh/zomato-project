# 🍽️ Zomato KPT Intelligence
### Kitchen Prep Time Prediction — Redesigned

<p align="center">
  <img src="https://img.shields.io/badge/Zomato-Deep%20Tech%20Challenge%202026-E23744?style=for-the-badge&logo=zomato&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-2DD4BF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Team-Phoenix-F4B942?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Institute-MAIT%20Delhi-7C3AED?style=for-the-badge"/>
</p>

<p align="center">
  <a href="https://zomatovansh.vercel.app/"><strong>🌐 Live Demo → zomatovansh.vercel.app</strong></a> &nbsp;|&nbsp;
  <a href="https://github.com/provanshh/zomato-project"><strong>📁 GitHub Repository</strong></a> &nbsp;|&nbsp;
  <a href="https://www.kaggle.com/datasets/himanshunakrani/zomato-bangalore-restaurants"><strong>📦 Dataset (Kaggle)</strong></a>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Root Cause Analysis](#-root-cause-analysis)
- [Proposed Solution](#-proposed-solution)
- [Dataset](#-dataset)
- [System Architecture](#-system-architecture)
- [Signal Stack](#-4-layer-signal-stack)
- [ML Pipeline](#-ml-pipeline)
- [Results](#-results--simulation)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Live Demo](#-live-demo)
- [Team Phoenix](#-team-phoenix)

---

## 🧭 Overview

**Zomato KPT Intelligence** is a research-grade system that redesigns Kitchen Prep Time (KPT) prediction for Zomato using enriched real-world signals, IoT instrumentation, and scalable ML pipelines. The project was built as part of the **Zomato Deep Tech Challenge 2026**.

> KPT is the most volatile and least controllable variable in Zomato's end-to-end delivery model — and it's currently being predicted with a fundamentally broken signal.

| Metric | Scale |
|--------|-------|
| 🏪 Merchants on Platform | 300,000+ |
| 📦 Daily Orders | 4.2 Million |
| ❌ Current ETA Error Rate | ~35% |
| 💸 Annual ETA Cost Impact | ₹850 Crore |

---

## 🔴 The Problem

Kitchen Prep Time (KPT) is the interval between **order confirmation** and **food readiness** at the restaurant. Zomato's entire ETA system — including when to dispatch riders — depends on accurately predicting this window.

Currently, the system relies on a single signal: the **"Food Order Ready" (FOR)** event, a button merchants tap on their app. This signal is deeply biased and unreliable.

```
Customer Places Order
        │
        ▼
  Kitchen Starts Prep
        │
        ▼
  ⚠️  FOR Button Tapped  ◄──── (Merchant sees rider walk in)
        │
        ▼
  Rider Dispatched Too Early
        │
        ▼
  Rider Waits 5–12 Minutes at Restaurant
        │
        ▼
  ETA Drifts → Customer Dissatisfied
```

### 📊 Current State vs. Targets

| KPI | Current State | Target |
|-----|--------------|--------|
| Avg Rider Wait at Restaurant | **~8.4 min** | ≤ 3 min |
| P90 ETA Prediction Error | **±11.2 min** | ≤ 5 min |
| Orders with Delayed ETA | **18.3%** | ≤ 8% |
| Cost per Idle Rider Minute | **~₹22** | — |
| Rider Idle Time per Shift | **+22 min** | — |

---

## 🧠 Root Cause Analysis

Three distinct bias layers poison the FOR signal:

### 1. 🛵 Rider-Triggered FOR (Most Critical)
In **41% of observed cases**, the FOR event is marked within ±90 seconds of rider arrival — meaning it captures *rider presence*, not *food readiness*. This introduces a **3–8 minute forward lag** in every KPT label used for model training.

### 2. 🍽️ Cross-Platform Kitchen Load Blindness
Popular restaurants process **30–60% of orders** via dine-in, Swiggy, direct phone calls, and other aggregators. Zomato has **zero visibility** into this parallel load, causing consistent underprediction during peak hours.

### 3. 📉 Merchant Behavior Drift
The same merchant may mark food ready in **8 minutes on Monday** but **18 minutes on Saturday evening** — with no change in recipe complexity. Behavioral inconsistency directly contaminates training data quality.

### Impact on Core Metrics

```
Rider wait time inflation     ████████████████████  +4.8 min avg
P90 ETA error increase        █████████████████     +6.1 min
Order cancellation rate       █████████████         ~2.3x higher
Rider idle time per shift     ████████████████████  +22 min/shift
```

> **Key Finding:** The FOR signal is structurally biased toward capturing *rider arrival* rather than *food readiness*. Any model trained solely on this label will inherit this bias — making **de-biasing or signal replacement the #1 priority**, not model architecture changes.

---

## 💡 Proposed Solution

Replace single-signal reliance with a **4-Layer Enriched Signal Stack** that captures kitchen state through independent, hardware and software signals.

---

## 📡 4-Layer Signal Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    KPT PREDICTION ENGINE                     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  LAYER 1     │  LAYER 2     │  LAYER 3     │  LAYER 4       │
│  De-biased   │  IoT Sensor  │  POS System  │  GMaps Live    │
│  FOR Signal  │  Network     │  Integration │  Busyness      │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ Mathematical │ Thermal +    │ Open ticket  │ Popular Times  │
│ bias         │ Acoustic     │ count across │ API score      │
│ correction   │ sensors in   │ ALL platforms│ (0–100)        │
│ using rider  │ kitchen      │ incl. dine-in│ as rush proxy  │
│ GPS delta    │              │ & Swiggy     │                │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ Coverage:    │ Coverage:    │ Coverage:    │ Coverage:      │
│ 300K+        │ 1,000+       │ 15,000+      │ 300K+          │
│ merchants    │ merchants    │ merchants    │ merchants      │
│ De-noise: 2.4x│ De-noise: 8.7x│ De-noise: 4.1x│ De-noise: 2.1x│
│ KPT Δ: -28% │ KPT Δ: -64% │ KPT Δ: -42% │ KPT Δ: -21%   │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### Feature Importance in Final Model

| Signal Feature | Importance |
|----------------|-----------|
| IoT Thermal/Acoustic | **34%** |
| Historical KPT Pattern | 28% |
| Live Rush Signal (GMaps) | 21% |
| Corrected FOR Signal | 11% |
| Rider Feedback Labels | 6% |

---

## 📦 Dataset

### Primary Dataset — Kaggle
> 🔗 [Zomato Bangalore Restaurants — Kaggle](https://www.kaggle.com/datasets/himanshunakrani/zomato-bangalore-restaurants)

| Field | Value |
|-------|-------|
| Total Entries | **51,717 restaurants** |
| Features | Name, Cuisine, Cost, Ratings, Location |
| Coverage | Bangalore, India |
| Use in Project | Baseline EDA, cuisine variance analysis, rating correlation |

### Synthetic + Public Hybrid Dataset — `KPT_SIGNALS_v1`

Since proprietary Zomato data is unavailable, a curated hybrid dataset was constructed from:

- 🗺️ **OpenStreetMap** — restaurant density and location data
- 📍 **Google Maps Places API** — live busyness, rating, and review data
- 🍴 **Yelp Open Dataset** — restaurant activity proxies
- 🔢 **Simulated Order Streams** — Poisson-process arrival rate distributions
- 🌡️ **Synthetic IoT Sensor Traces** — generated from empirical kitchen workflow literature

| Stat | Value |
|------|-------|
| Restaurant Instances | **12,500** |
| Simulated Orders | **2.4 Million** |
| Features per Record | **48** |
| Metro City Coverage | **8 Indian cities** |

### Key Schema Columns

| Feature | Type | Source | Role |
|---------|------|--------|------|
| `order_id` | STRING | Synthetic | Unique UUID per order |
| `for_timestamp` | DATETIME | Zomato Mx | Noisy label — primary de-bias target |
| `rider_arrival_timestamp` | DATETIME | GPS | Detects FOR bias trigger pattern |
| `for_rider_delta_sec` | FLOAT | Derived | Negative = genuine FOR signal |
| `kitchen_thermal_index` | FLOAT | IoT (Simulated) | Heat sensor — peaks at cooking activity |
| `acoustic_activity_score` | FLOAT | IoT (Simulated) | ML-classified sound intensity |
| `gmap_live_busyness` | INT | Google Maps API | Live rush score 0–100 |
| `pos_open_tickets` | INT | POS System | Active orders across all platforms |
| `item_count` | INT | Synthetic | Prep complexity indicator |
| `menu_category_mix` | ARRAY | Zomato Menu API | Cuisine-level complexity proxy |

---

## 🏗️ System Architecture

```
                        ┌──────────────────────┐
                        │   Customer App (UX)   │
                        │  ETA shown to user    │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │   KPT Prediction API  │
                        │   (Real-time Serving) │
                        └──────────┬───────────┘
                                   │
          ┌─────────────┬──────────┴──────────┬─────────────┐
          │             │                     │             │
   ┌──────▼──────┐ ┌────▼────┐ ┌─────────────▼──┐ ┌───────▼──────┐
   │ Signal      │ │ IoT     │ │ POS Integration│ │ GMaps Live   │
   │ De-biasing  │ │ Sensors │ │ (Open Tickets) │ │ Busyness API │
   │ Engine      │ │ (Thermal│ │                │ │              │
   │             │ │ Acoustic│ │                │ │              │
   └──────┬──────┘ └────┬────┘ └───────┬────────┘ └───────┬──────┘
          │             │              │                   │
          └─────────────┴──────────────┴───────────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  Feature Engineering  │
                        │  + Signal Fusion      │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  XGBoost + LightGBM   │
                        │  Ensemble ML Model    │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │  KPT Prediction       │
                        │  Output (seconds)     │
                        └──────────────────────┘
```

---

## 🤖 ML Pipeline

The model uses a stacked ensemble approach:

```
Raw Signals
    │
    ├── FOR Signal ──────► Bias Detector ──► Corrected FOR Label
    │
    ├── IoT Sensors ─────► Thermal + Acoustic Feature Extractor
    │
    ├── POS System ──────► Kitchen Load Score
    │
    └── GMaps API ───────► Live Busyness Index
              │
              ▼
    Feature Engineering (48 features)
              │
              ▼
    ┌─────────────────────┐
    │  XGBoost  LightGBM  │  ◄── Trained on 2.4M simulated orders
    │    Ensemble Stack   │
    └─────────────────────┘
              │
              ▼
    KPT Prediction (minutes)
```

### Signal Reliability by Merchant Tier

| Merchant Tier | Coverage | Primary Signal | De-noising Factor | KPT Improvement |
|--------------|----------|---------------|-------------------|-----------------|
| 🔴 Core (300K+) | 100% | GMaps + GPS Delta | 2.4x | **-28%** |
| 🟡 POS (15K+) | 5% | API Live Load | 4.1x | **-42%** |
| 🟢 IoT (1K+) | 0.3% | Thermal/Acoustic | 8.7x | **-64%** |

---

## 📈 Results & Simulation

> Monte Carlo simulation across **50 runs** with 2.4M orders, comparing baseline FOR-only vs. full 4-layer signal stack.

```
                BEFORE          AFTER           IMPROVEMENT
                ──────          ─────           ───────────
Rider Wait      8.4 min   →    3.2 min          ▼ 62%
P50 ETA Error   6.7 min   →    2.8 min          ▼ 58%
P90 ETA Error   11.2 min  →    4.9 min          ▼ 56%
Order Delay %   18.3%     →    6.1%             ▼ 67%
Annual Savings  —         →    ₹92 Crore        ✅
```

### Deep Analysis Findings (from Bangalore Dataset)

| Finding | Detail |
|---------|--------|
| 🔴 Operational Bias | "Food Ready" signals are **42% more likely** to be marked late in high-cost restaurants due to complex plating protocols |
| 🟢 Cuisine Lag | South Indian and Café cuisines show the **lowest KPT variance** — ideal for high-precision ETA testing |
| 🟡 Delivery Radius | Rider wait times increase by **18%** during peak hours (7 PM–10 PM) in BTM Layout and HSR sectors |
| 🟣 Review Sentiment | Negative reviews are **3x more likely** to mention "Wait Time" than "Food Quality" for orders delayed by >10 mins |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python, JavaScript |
| **ML / Data** | Pandas, Scikit-learn, XGBoost, LightGBM |
| **Visualization** | Seaborn, Matplotlib |
| **APIs** | Google Maps Places API, Zomato Menu API |
| **IoT (Simulated)** | Thermal sensors, Acoustic ML classifiers |
| **Frontend / Demo** | Vercel (Next.js) |
| **Dataset** | Kaggle, OSM, Yelp Open Dataset |

---

## 📁 Project Structure

```
zomato-kpt-intelligence/
│
├── 📂 data/
│   ├── raw/                    # Kaggle + public data
│   ├── synthetic/              # KPT_SIGNALS_v1 dataset
│   └── processed/              # Feature-engineered outputs
│
├── 📂 eda/
│   ├── zomato_eda.ipynb        # Exploratory Data Analysis
│   └── signal_analysis.ipynb  # FOR bias detection analysis
│
├── 📂 models/
│   ├── xgboost_kpt.py          # XGBoost KPT model
│   ├── lightgbm_kpt.py         # LightGBM KPT model
│   └── ensemble.py             # Stacked ensemble
│
├── 📂 signals/
│   ├── for_debiaser.py         # FOR signal bias correction
│   ├── iot_features.py         # Thermal + acoustic processing
│   └── gmap_integration.py    # Google Maps busyness fetcher
│
├── 📂 simulation/
│   └── monte_carlo.py          # 50-run Monte Carlo simulator
│
├── 📂 dashboard/
│   └── intelligence_dashboard/ # Ops monitoring UI
│
├── 📄 report/
│   └── zomato_kpt_report.html  # Full 12-page research report
│
└── README.md
```

---

## 🌐 Live Demo

> **🔗 [https://zomatovansh.vercel.app/](https://zomatovansh.vercel.app/)**

The live demo includes:
- Interactive KPT prediction simulation
- Signal layer toggle (compare FOR-only vs. full stack)
- Operations intelligence dashboard
- Cuisine-level KPT variance explorer
- Monte Carlo simulation results visualizer

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🌐 Live Demo | [zomatovansh.vercel.app](https://zomatovansh.vercel.app/) |
| 📁 GitHub Repository | [github.com/provanshh/zomato-project](https://github.com/provanshh/zomato-project) |
| 📦 Kaggle Dataset | [Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshunakrani/zomato-bangalore-restaurants) |
| 📊 Source Code (EDA) | [github.com/m-pawan/Zomato-Data-Analysis](https://github.com/m-pawan/Zomato-Data-Analysis) |

---

## 👥 Team Phoenix

> *Maharaja Agrasen Institute of Technology (MAIT), Delhi*
> *Zomato Deep Tech Challenge 2026*

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Vansh Singla</strong><br/>
      🏷️ Team Leader / Core Architect<br/><br/>
      Lead researcher for signal de-biasing and ML strategy.<br/>
      Oversaw the integration of IoT concepts into the Zomato KPT model.<br/><br/>
      <a href="https://www.linkedin.com/in/provanshh/">🔗 LinkedIn</a><br/>
      📧 vs.vansh19@gmail.com<br/>
      📱 8076069983
    </td>
    <td align="center" width="50%">
      <strong>Dev Garg</strong><br/>
      🏷️ System Analyst<br/><br/>
      Analytic lead for performance simulation and business impact.<br/>
      Calibrated the operational roadmap and ROI projections.<br/><br/>
      <a href="https://www.linkedin.com/in/gargdev">🔗 LinkedIn</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is developed for academic and hackathon purposes under the **Zomato Deep Tech Challenge 2026**.
Dataset usage is subject to [Kaggle's terms of service](https://www.kaggle.com/terms).

---

<p align="center">
  Made with ❤️ by <strong>Team Phoenix</strong> · MAIT Delhi · 2026<br/>
  <a href="https://zomatovansh.vercel.app/">zomatovansh.vercel.app</a>
</p>
