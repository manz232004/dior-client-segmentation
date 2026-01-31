"""
Segment customers with KMeans; choose k by silhouette; assign persona names from rules.
Reads data/customer_features.csv, writes data/segmented_customers.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_features(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "customer_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/build_features.py first.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("customer_features.csv is empty.")
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df["log_monetary_total"] = np.log1p(df["monetary_total"])
    df["log_avg_basket_value"] = np.log1p(df["avg_basket_value"])

    feature_cols = [
        "recency_days",
        "frequency_orders",
        "log_monetary_total",
        "log_avg_basket_value",
        "discount_rate",
        "boutique_rate",
    ]
    share_cols = [c for c in df.columns if c.startswith("category_share_")]
    feature_cols = feature_cols + share_cols
    X = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index), feature_cols, scaler


def choose_k(X: pd.DataFrame, k_range: range = range(4, 9)) -> int:
    from sklearn.metrics import silhouette_score
    best_k, best_score = 4, -1.0
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _safe(val, default=0):
    if pd.isna(val):
        return default
    return float(val)


def assign_persona_for_cluster(med: pd.Series) -> str:
    monetary = _safe(med.get("monetary_total", 0))
    freq = _safe(med.get("frequency_orders", 0))
    recency = _safe(med.get("recency_days", 999))
    boutique = _safe(med.get("boutique_rate", 0))
    beauty = _safe(med.get("category_share_Beauty", 0))
    accessories = _safe(med.get("category_share_Accessories", 0))
    rtw = _safe(med.get("category_share_RTW", 0))

    if monetary > 2000 and boutique > 0.5:
        return "Couture Connoisseurs"
    if freq >= 4 and beauty > 0.35:
        return "Beauty Regulars"
    if accessories > 0.4:
        return "Accessory Collectors"
    if recency < 90 and monetary < 800:
        return "Trend-Driven Newcomers"
    if rtw > 0.35:
        return "Ready-to-Wear Focus"
    if monetary > 1500:
        return "High-Value Clients"
    if freq >= 3:
        return "Loyal Shoppers"
    return "Explorers"


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = load_features(data_dir)

    X_scaled, feature_cols, scaler = preprocess(df)
    k = choose_k(X_scaled)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(X_scaled)

    # One persona per cluster from cluster medians
    cluster_medians = df.groupby("cluster_id")[
        ["monetary_total", "frequency_orders", "recency_days", "boutique_rate"]
        + [c for c in df.columns if c.startswith("category_share_")]
    ].median()

    persona_map = {
        cid: assign_persona_for_cluster(cluster_medians.loc[cid])
        for cid in df["cluster_id"].unique()
    }
    df["persona_name"] = df["cluster_id"].map(persona_map)

    out_path = data_dir / "segmented_customers.csv"
    df.to_csv(out_path, index=False)
    print(f"Chose k={k}. Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
