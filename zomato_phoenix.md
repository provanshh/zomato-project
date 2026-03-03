# Zomato KPT Intelligence — Technical Assets & Research Repository

This document contains all technical links, simulation statistics, and research methodologies used in the **Zomato KPT (Kitchen Prep Time) Intelligence Report 2026**.

## 📊 Core Performance Metrics (Simulated v1.0)
Based on a Monte Carlo simulation of 2.4M orders across 300,000 merchants.

| Metric | Baseline (FOR-Only) | Intelligence System (4-Layer) | Improvement |
|:---|:---|:---|:---|
| **Avg Rider Wait Time** | 8.4 min | 3.2 min | **-62%** |
| **P50 ETA Error (MAE)** | 6.7 min | 2.8 min | **-58%** |
| **P90 ETA Error** | 11.2 min | 4.9 min | **-56%** |
| **Order Delay Rate** | 18.3% | 6.1% | **-67%** |
| **Cancellation Rate (Late)**| 4.7% | 1.6% | **-66%** |
| **Rider Earnings / Hr** | Base | +16.4% | **Significant Increase** |

---

## 🏗️ Signal Layer Sensitivity
How different signal inputs contribute to final KPT prediction accuracy:

| Signal Layer | Type | Model Sensitivity | Primary Methodology |
|:---|:---|:---|:---|
| **IoT State Features** | Thermal / Acoustic | **93%** | Hidden Markov Model (HMM) |
| **Live Rush Signals** | Google Maps / POS | **81%** | API Real-time Scraping |
| **Historical Patterns** | Temporal (7d/30d) | **76%** | LightGBM Rolling Averaging |
| **FOR De-bias Score** | GPS Delta Analysis| **85%** | Centroid-Movement Logic |

---

## 🔗 Repository & Dataset Links

### 🛠️ GitHub Repositories
*   **[Main Intelligence Hub](https://github.com/provanshh/zomato-project)** - *Core logic & Signal Engineering*
*   [IoT Stream Adapter](https://github.com/provanshh/zomato-project/tree/main/iot-adapter) - *Thermal flux analysis*
*   [Signal De-noising Pipeline](https://github.com/provanshh/zomato-project/tree/main/denoise) - *GPS Centroid movement*
*   [Monte Carlo Sim Engine](https://github.com/provanshh/zomato-project/tree/main/simulation) - *Performance modeling*
*   [Merchant Dashboard API](https://github.com/provanshh/zomato-project/tree/main/api) - *Restaurant Ops Interface*

### 📊 Datasets & Feeds
*   **[Zomato Bangalore Corpus (Kaggle)](https://www.kaggle.com/datasets/himanshunakrani/zomato-bangalore-restaurants)** - *Primary merchant dataset*
*   [KPT_SIGNALS_v1.csv](https://github.com/provanshh/zomato-project/data/signals_v1.zip) - *2.4M Simulated orders*
*   [Thermal Kitchen Benchmarks](https://github.com/provanshh/zomato-project/data/thermal_bench.json) - *IoT training data*
*   [Rider Wait Time Delta Set](https://github.com/provanshh/zomato-project/data/rider_deltas.csv) - *Historical wait logs*

---

## 🔬 Research & Methodology
1.  **Stochastic Modeling**: Monte Carlo approach decomposing delivery pipelines into 12,000 distinct probability nodes.
2.  **Kitchen State Inference**: Hidden Markov Models calibrated via commercial kitchen workflow studies.
3.  **Cross-Market Synthesis**: Integration of Google Places "Live Busyness" with internal merchant load.
4.  **Edge ML Deployment**: Constraints-aware optimization for high-density delivery lattices in Tier-1 metros.

---
**Prepared by Team Phoenix**  
*Maharaja Agrasen Institute of Technology, Delhi*  
*Contact: vs.vansh19@gmail.com*
