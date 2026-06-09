# 🎨 View TCRPPO v2 Training Traces

## 📊 Quick View

All visualizations are in `logs/` directory.

### 🖼️ Main Visualizations

1. **[logs/alive_traces_affinity_curves.png](logs/alive_traces_affinity_curves.png)** - All traces comparison
2. **[logs/alive_traces_statistics.png](logs/alive_traces_statistics.png)** - Statistical summary

### 🔍 Detailed Analysis (Best Traces)

- **[logs/trace61_detailed_analysis.png](logs/trace61_detailed_analysis.png)** - Best: 1.050, Avg: -1.392, 72.3% positive Δ ⭐
- **[logs/trace72_detailed_analysis.png](logs/trace72_detailed_analysis.png)** - Best: 0.747, Avg: -1.238, 65.2% positive Δ
- **[logs/trace73_detailed_analysis.png](logs/trace73_detailed_analysis.png)** - Best: 0.834, Avg: -1.184, 65.9% positive Δ
- **[logs/trace70_detailed_analysis.png](logs/trace70_detailed_analysis.png)** - Best: 0.702, Avg: -1.216, 68.8% positive Δ

## 🚀 Generate New Visualizations

```bash
cd /share/liuyutian/tcrppo_v2

# Quick commands
./quick_viz.sh all      # All traces
./quick_viz.sh recent   # Last 10 traces
./quick_viz.sh best     # Best traces (61,70,72,73)
./quick_viz.sh 72       # Specific trace

# Advanced
python visualize_alive_traces.py --specific-traces 61 72 73
python visualize_single_trace.py 72
```

## 📖 Documentation

- **[logs/README_VISUALIZATION.md](logs/README_VISUALIZATION.md)** - Complete guide
- **[logs/VISUALIZATION_REPORT.md](logs/VISUALIZATION_REPORT.md)** - Detailed analysis report

## 🏆 Top 5 Traces

| Trace | Best Affinity | Avg Affinity | Episodes | Status |
|-------|---------------|--------------|----------|--------|
| trace61 | 1.050 ⭐ | -1.392 | 4,208 | Dynamic pool |
| trace54 | 0.938 | -1.945 | 10,328 | Hybrid abs delta |
| trace73 | 0.834 | -1.184 | 4,240 | Curriculum exploration |
| trace53 | 0.806 | -1.797 | 27,816 | Terminal reward |
| trace72 | 0.747 | -1.238 | 13,232 | Delta from trace70 |

## 💡 Quick Stats

- **Total traces analyzed:** 18
- **Total episodes:** 108,000+
- **Best peak affinity:** 1.050 (trace61)
- **Longest run:** 27,816 episodes (trace53)
- **Best improvement rate:** 72.3% positive deltas (trace61)

---

**Last updated:** 2026-05-29
