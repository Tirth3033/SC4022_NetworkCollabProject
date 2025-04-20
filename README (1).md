
# 📘 Collaboration Network Analysis from DBLP

This project builds and analyzes a co-authorship network using DBLP author profiles. It processes an input Excel file of data scientists, fetches their co-authorship records from DBLP, and constructs a graph to study collaboration patterns, network evolution, and diversity.

---

## 📁 Project Directory Structure

```
project/
├── project.py                 # Main script (this file)
├── Input/
│   ├── scientists.xlsx        # Excel file containing DBLP links/IDs and optional metadata
│   └── config.txt             # (Optional) Contains a number for `kmax` (e.g., 5)
├── Results/
│   └── *.png                  # Output visualizations saved here
```

---

## ✅ Features

- Builds a real-world co-authorship graph using DBLP data
- Analyzes structural properties like degree, clustering, hubs, and components
- Tracks the temporal evolution of the network
- Compares with a random network model
- Applies a transformation to limit node degrees while maximizing diversity
- Plots and reports key insights with graphs

---

## 🛠️ Requirements

Install all required packages using:

```bash
pip install pandas requests beautifulsoup4 lxml networkx matplotlib xlrd openpyxl
```

### 📦 Why these packages?
- `pandas`: For reading and handling Excel files
- `requests`: To download data from DBLP
- `beautifulsoup4` + `lxml`: For parsing XML from DBLP
- `networkx`: For graph construction and analysis
- `matplotlib`: For visualizations
- `xlrd`, `openpyxl`: To support both `.xls` and `.xlsx` formats

---

## 📥 Input File Format

Your Excel file (placed in `Input/`) must include at least one column with **DBLP profile links or IDs**.

### Example Columns:
- `dblp_url` or `dblp_id_url` (Required)
- `name` (Optional but recommended)
- `country` (Optional)
- `institution` (Optional)
- `expertise` (Optional: values 1–10)

> The script auto-detects the DBLP column, and will randomly assign expertise if it's missing.

---

## ⚙️ Config File (Optional)

Create a file named `config.txt` in the `Input/` folder to specify `kmax` for transformation (max degree per node):

```
kmax = 7
```

If omitted, the script defaults to `kmax = 5`.

---

## ▶️ How to Run

Navigate to the folder containing `project.py` and run:

```bash
python project.py
```

Make sure:
- `Input/` folder exists
- Excel file is present and correctly formatted
- Internet connection is available (for DBLP XML fetches)

---

## 📊 Output

All output plots and comparisons are saved in the `Results/` folder.

### Generated Files:
- `degree_distribution.png`: Histogram of degrees (real network)
- `degree_compare_real_random.png`: Degree distribution – real vs random
- `degree_compare_transformed.png`: Degree distribution – real vs transformed
- `real_vs_random_metrics.png`: Metrics (components, clustering, etc.) comparison
- `connectivity_compare_transformed.png`: Before vs after transformation

---

## 📈 Example Output (Based on 1070 Scientists)

**Real Network:**
- Nodes: 1070, Edges: 7298
- Max Degree: 94, Avg Clustering: 0.2890
- Largest Component Size: 961
- Isolated Nodes: 99 (9.3%)
- Top Hubs: Divesh Srivastava, Beng Chin Ooi, etc.

**Random Network:**
- Nodes: 1070, Edges: 7298
- Max Degree: 26
- Avg Clustering: 0.0119
- Fully connected

**Transformed Network (kmax = 5):**
- Edges reduced to 1404
- Isolates increased to 177
- Minor loss in diversity (countries/institutions)

---

## 🧠 Analysis Summary

The project provides insights into:
- Network centrality and clustering
- Hubs and isolated authors
- How collaborations grew from 1980s to 2025
- How limiting collaborations affects connectivity and diversity

---

## 🌐 Notes

- Ensure good internet connection – DBLP XML pages are fetched live.
- If DBLP returns error codes (e.g., 404/410), those scientists are skipped.
- Time to run: 10–30 mins depending on system and network.

---

## 🤝 Credits

Built using:
- [DBLP](https://dblp.org/) for scholarly data
- Python ecosystem (NetworkX, matplotlib, pandas)

---

## 📬 Support

If the script fails:
- Double-check Excel formatting and DBLP links
- Ensure all Python packages are installed
- Make sure XML parsing works: `pip install lxml beautifulsoup4`
