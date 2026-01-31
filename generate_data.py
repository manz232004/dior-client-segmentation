"""
Generate synthetic Dior-like customer and order data.
Luxury assumptions: lognormal spend, boutique/mixed higher basket, category mix by channel, rare discounts.
"""
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_customers(n_customers: int = 2500) -> pd.DataFrame:
    regions = ["EMEA", "Americas", "Asia-Pacific", "Japan"]
    channels = ["Online", "Boutique", "Mixed"]
    acquisition = ["Boutique", "Online", "Social", "Referral", "Event"]
    age_bands = ["18-24", "25-34", "35-44", "45-54", "55+"]
    tiers = ["Standard", "Loyalty", "VIP"]

    rows = []
    for i in range(1, n_customers + 1):
        rows.append({
            "customer_id": f"C{i:06d}",
            "region": random.choice(regions),
            "preferred_channel": random.choice(channels),
            "acquisition_source": random.choice(acquisition),
            "age_band": random.choice(age_bands),
            "tier": random.choice(tiers),
        })
    return pd.DataFrame(rows)


def generate_orders(
    customers_df: pd.DataFrame,
    months: int = 18,
    orders_per_customer_mean: float = 4.5,
    orders_per_customer_std: float = 3.0,
) -> pd.DataFrame:
    categories = ["Beauty", "Fragrance", "Accessories", "RTW"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)

    order_id = 0
    rows = []

    for _, cust in customers_df.iterrows():
        cid = cust["customer_id"]
        channel = cust["preferred_channel"]
        n_orders = max(0, int(np.random.normal(orders_per_customer_mean, orders_per_customer_std)))
        n_orders = min(n_orders, 50)

        for _ in range(n_orders):
            order_id += 1
            # Uniform random date in range
            delta = (end_date - start_date).days
            days_ago = random.randint(0, delta)
            order_date = end_date - timedelta(days=days_ago)

            # Channel affects category mix: Beauty more online; RTW/Accessories more boutique
            if channel == "Online":
                probs = [0.45, 0.25, 0.15, 0.15]  # Beauty, Fragrance, Accessories, RTW
            elif channel == "Boutique":
                probs = [0.15, 0.20, 0.35, 0.30]
            else:  # Mixed
                probs = [0.30, 0.25, 0.25, 0.20]
            category = np.random.choice(categories, p=probs)

            # Boutique/Mixed higher basket than Online; lognormal base
            base_mean = 180 if channel == "Online" else 320 if channel == "Boutique" else 280
            base_std = 0.8
            basket_value = np.random.lognormal(np.log(base_mean), base_std)
            basket_value = max(50, min(5000, basket_value))
            basket_value = round(basket_value, 2)

            items_count = max(1, int(np.random.poisson(3) + 1))
            if category in ["Accessories", "RTW"]:
                items_count = max(1, min(items_count, 4))

            # Discounts rare in luxury
            discount_flag = 1 if random.random() < 0.12 else 0
            # Boutique flag: higher for Boutique/Mixed
            p_boutique = 0.85 if channel == "Boutique" else (0.45 if channel == "Mixed" else 0.08)
            boutique_flag = 1 if random.random() < p_boutique else 0

            rows.append({
                "order_id": f"ORD{order_id:07d}",
                "customer_id": cid,
                "order_date": order_date.strftime("%Y-%m-%d"),
                "category": category,
                "basket_value": basket_value,
                "items_count": items_count,
                "discount_flag": discount_flag,
                "boutique_flag": boutique_flag,
            })

    return pd.DataFrame(rows)


def main() -> None:
    set_seed(42)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_customers = 2500
    months = 18

    print("Generating customers...")
    customers_df = generate_customers(n_customers=n_customers)
    customers_path = out_dir / "customers.csv"
    customers_df.to_csv(customers_path, index=False)
    print(f"Saved {len(customers_df)} customers to {customers_path}")

    print("Generating orders...")
    orders_df = generate_orders(customers_df, months=months)
    orders_path = out_dir / "orders.csv"
    orders_df.to_csv(orders_path, index=False)
    print(f"Saved {len(orders_df)} orders to {orders_path}")

    print("Done.")


if __name__ == "__main__":
    main()
