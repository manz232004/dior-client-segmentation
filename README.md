# Dior-Style Customer Segmentation

A synthetic customer segmentation project built to mimic luxury retail patterns (Dior-like). It generates realistic transactional data, engineers RFM and category-mix features, clusters customers with KMeans, assigns persona labels, and surfaces everything in a clean black-and-white Streamlit dashboard.

**Note:** This uses a **synthetic dataset**—no real customer data. Assumptions (skewed spend, channel mix, rare discounts) are tuned to reflect luxury behaviour.

## Setup

```bash
pip install -r requirements.txt
```

## Run End-to-End

1. **Generate synthetic data** (customers + orders):
   ```bash
   python src/generate_data.py
   ```
   Creates `data/customers.csv` and `data/orders.csv` (default: 2500 customers, 18 months).

2. **Build features** (RFM + category shares):
   ```bash
   python src/build_features.py
   ```
   Creates `data/customer_features.csv`.

3. **Segment customers** (KMeans + persona names):
   ```bash
   python src/segment_customers.py
   ```
   Creates `data/segmented_customers.csv`.

4. **Launch dashboard**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `src/generate_data.py` — Synthetic customers and orders with luxury assumptions
- `src/build_features.py` — One row per customer: RFM, category shares, channel mix
- `src/segment_customers.py` — Clustering and persona labeling
- `app.py` — Streamlit app: KPIs, cluster table, charts, customer explorer
- `data/` — Generated CSVs (created by the scripts above)

## Deploy on Streamlit Cloud

1. **Push this repo to GitHub**
   - Create a new repository on [GitHub](https://github.com/new) (e.g. `dior-client-segmentation`).
   - Run in this folder (replace `YOUR_USERNAME` and `REPO_NAME` with your GitHub username and repo name):

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
   - Click **New app**, select this repository, set **Main file path** to `app.py`.
   - Click **Deploy**. The app will run using the committed `data/` files.

## Dependencies

- pandas, numpy, scikit-learn, plotly, streamlit
