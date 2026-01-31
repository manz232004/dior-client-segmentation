"""
Build one row per customer: RFM, category shares, channel mix.
Reads data/customers.csv and data/orders.csv, writes data/customer_features.csv.
"""
from pathlib import Path

import pandas as pd


def load_data(data_dir: Path):
    customers_path = data_dir / "customers.csv"
    orders_path = data_dir / "orders.csv"
    if not customers_path.exists():
        raise FileNotFoundError(f"Missing {customers_path}. Run src/generate_data.py first.")
    if not orders_path.exists():
        raise FileNotFoundError(f"Missing {orders_path}. Run src/generate_data.py first.")
    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    if customers.empty or orders.empty:
        raise ValueError("Customers or orders data is empty.")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    return customers, orders


def build_features(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    max_date = orders["order_date"].max()

    # Order-level aggregates per customer
    agg = orders.groupby("customer_id").agg(
        order_count=("order_id", "nunique"),
        monetary_total=("basket_value", "sum"),
        recency_days=("order_date", lambda x: (max_date - x.max()).days),
        avg_basket_value=("basket_value", "mean"),
        avg_items_count=("items_count", "mean"),
        discount_rate=("discount_flag", "mean"),
        boutique_rate=("boutique_flag", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"order_count": "frequency_orders"})

    # Category spend shares
    category_pivot = orders.pivot_table(
        index="customer_id",
        columns="category",
        values="basket_value",
        aggfunc="sum",
        fill_value=0,
    )
    total_spend = category_pivot.sum(axis=1)
    for col in category_pivot.columns:
        category_pivot[f"category_share_{col}"] = (
            category_pivot[col] / total_spend.replace(0, 1)
        )
    share_cols = [c for c in category_pivot.columns if c.startswith("category_share_")]
    category_pivot = category_pivot[share_cols].reset_index()

    out = customers[["customer_id"]].merge(agg, on="customer_id", how="left")
    out = out.merge(category_pivot, on="customer_id", how="left")
    out = out.fillna(0)
    return out


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    customers, orders = load_data(data_dir)
    features = build_features(customers, orders)
    out_path = data_dir / "customer_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features)} rows to {out_path}")


if __name__ == "__main__":
    main()
