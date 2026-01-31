"""
Dior-style Customer Segmentation Dashboard.
Matches Dior UI: white background, black text, elegant serif headings, minimal containers.
Interactive Plotly charts. No extra rectangular bars.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"
SEGMENTED_PATH = DATA_DIR / "segmented_customers.csv"

st.set_page_config(page_title="Client Segmentation | Dior", layout="wide", initial_sidebar_state="collapsed")

# Dior-style: white bg, black text, serif headings, no boxy containers
DIOR_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp { background: #ffffff !important; }
    [data-testid="stAppViewContainer"] { background: #ffffff !important; max-width: 1200px; margin: 0 auto; padding: 2rem 3rem; }
    [data-testid="stHeader"] { background: #ffffff !important; border-bottom: 1px solid #e5e5e5 !important; }
    
    /* No boxes around blocks or columns */
    [data-testid="stVerticalBlock"] > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    div[data-testid="column"] > div { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0.5rem 0 !important; }
    
    h1, h2, h3 { color: #000000 !important; font-family: 'Cormorant Garamond', Georgia, serif !important; font-weight: 600 !important; }
    .stMarkdown { color: #000000 !important; font-family: 'Inter', -apple-system, sans-serif !important; }
    p, label, span { color: #1a1a1a !important; font-family: 'Inter', sans-serif !important; }
    
    .stMetric label { color: #666666 !important; font-family: 'Inter', sans-serif !important; font-weight: 500 !important; letter-spacing: 0.04em !important; text-transform: uppercase !important; font-size: 0.7rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #000000 !important; font-family: 'Cormorant Garamond', serif !important; font-weight: 600 !important; }
    
    /* Tables: minimal, no heavy box */
    [data-testid="stDataFrame"] { border: 1px solid #e5e5e5 !important; border-radius: 0 !important; overflow: hidden !important; }
    [data-testid="stDataFrame"] table { background: #ffffff !important; color: #000000 !important; }
    [data-testid="stDataFrame"] th { background: #fafafa !important; color: #000000 !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; border-bottom: 1px solid #e5e5e5 !important; }
    [data-testid="stDataFrame"] td { color: #1a1a1a !important; border-color: #eee !important; font-family: 'Inter', sans-serif !important; }
    [data-testid="stDataFrame"] tr:hover td { background: #f9f9f9 !important; }
    
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e5e5e5 !important; }
    [data-testid="stSelectbox"] label { color: #1a1a1a !important; }
    .stSelectbox div[data-baseweb="select"] { background: #ffffff !important; border: 1px solid #ccc !important; color: #000000 !important; border-radius: 0 !important; }
    
    hr { border-color: #e5e5e5 !important; }
    .dior-title { font-family: 'Cormorant Garamond', Georgia, serif !important; font-size: 2.5rem !important; font-weight: 600 !important; letter-spacing: 0.06em !important; color: #000 !important; text-align: center !important; margin-bottom: 0.25rem !important; }
    .dior-subtitle { font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important; color: #666 !important; text-align: center !important; letter-spacing: 0.03em !important; margin-bottom: 2rem !important; }
    .section-head { font-family: 'Cormorant Garamond', serif !important; font-size: 1.35rem !important; font-weight: 600 !important; color: #000 !important; margin-top: 2.5rem !important; margin-bottom: 1rem !important; }
</style>
"""

# Plotly: Dior B&W, interactive
PLOTLY_CONFIG = {"displayModeBar": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "responsive": True}


def plotly_dior_layout():
    return dict(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#fafafa",
        font=dict(family="Inter, sans-serif", color="#1a1a1a", size=12),
        title_font=dict(family="Cormorant Garamond, serif", color="#000000", size=18),
        xaxis=dict(gridcolor="#eee", zerolinecolor="#e5e5e5", linecolor="#ccc", tickfont=dict(color="#333")),
        yaxis=dict(gridcolor="#eee", zerolinecolor="#e5e5e5", linecolor="#ccc", tickfont=dict(color="#333")),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color="#333", size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=50, l=60, r=30),
        hovermode="x unified",
        showlegend=True,
    )


@st.cache_data
def load_segmented() -> pd.DataFrame:
    if not SEGMENTED_PATH.exists():
        st.error("Segmented data not found. Run: `python src/generate_data.py` then `python src/build_features.py` then `python src/segment_customers.py`.")
        st.stop()
    df = pd.read_csv(SEGMENTED_PATH)
    if df.empty:
        st.error("Segmented data is empty.")
        st.stop()
    return df


def main():
    st.markdown(DIOR_CSS, unsafe_allow_html=True)
    df = load_segmented()

    # --- Header (no extra bars) ---
    st.markdown('<p class="dior-title">Client Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="dior-subtitle">Synthetic luxury cohort · RFM & category mix</p>', unsafe_allow_html=True)

    # --- KPIs (no box styling; metrics only) ---
    n_customers = len(df)
    clv_proxy = df["monetary_total"].mean()
    n_clusters = df["cluster_id"].nunique()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total customers", f"{n_customers:,}")
    with col2:
        st.metric("Avg CLV proxy (total spend)", f"€{clv_proxy:,.0f}")
    with col3:
        st.metric("Segments", n_clusters)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Segment overview ---
    st.markdown('<p class="section-head">Segment overview</p>', unsafe_allow_html=True)
    cluster_agg = df.groupby("persona_name").agg(
        size=("customer_id", "count"),
        avg_spend=("monetary_total", "mean"),
        avg_recency_days=("recency_days", "mean"),
        avg_boutique_rate=("boutique_rate", "mean"),
    ).reset_index()
    cluster_agg.columns = ["Persona", "Size", "Avg spend (€)", "Avg recency (days)", "Boutique rate"]
    cluster_agg["Avg spend (€)"] = cluster_agg["Avg spend (€)"].round(0)
    cluster_agg["Avg recency (days)"] = cluster_agg["Avg recency (days)"].round(0)
    cluster_agg["Boutique rate"] = (cluster_agg["Boutique rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(cluster_agg, use_container_width=True, hide_index=True)

    # --- Interactive charts ---
    st.markdown('<p class="section-head">Spend by persona</p>', unsafe_allow_html=True)
    spend_by_persona = df.groupby("persona_name")["monetary_total"].mean().reset_index(name="avg_spend")
    fig_spend = px.bar(
        spend_by_persona, x="persona_name", y="avg_spend", text_auto=".0f",
        color="avg_spend", color_continuous_scale=["#f0f0f0", "#1a1a1a"],
        labels=dict(persona_name="", avg_spend="Avg spend (€)"),
    )
    fig_spend.update_layout(**plotly_dior_layout())
    fig_spend.update_traces(textfont_color="#000", textposition="outside", marker_line_color="#ccc", marker_line_width=1, showlegend=False)
    fig_spend.update_coloraxes(showscale=False)
    fig_spend.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig_spend, use_container_width=True, config=PLOTLY_CONFIG)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-head">Recency by persona</p>', unsafe_allow_html=True)
        rec_by_persona = df.groupby("persona_name")["recency_days"].mean().reset_index(name="avg_recency")
        fig_rec = px.bar(
            rec_by_persona, x="persona_name", y="avg_recency", text_auto=".0f",
            color="avg_recency", color_continuous_scale=["#1a1a1a", "#e0e0e0"],
            labels=dict(persona_name="", avg_recency="Avg recency (days)"),
        )
        fig_rec.update_layout(**plotly_dior_layout())
        fig_rec.update_traces(textfont_color="#000", textposition="outside", marker_line_color="#ccc", marker_line_width=1, showlegend=False)
        fig_rec.update_coloraxes(showscale=False)
        fig_rec.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_rec, use_container_width=True, config=PLOTLY_CONFIG)
    with col_b:
        st.markdown('<p class="section-head">Category share by persona</p>', unsafe_allow_html=True)
        cat_cols = [c for c in df.columns if c.startswith("category_share_")]
        if cat_cols:
            cat_long = df.groupby("persona_name")[cat_cols].mean().reset_index()
            cat_long = cat_long.set_index("persona_name").stack().reset_index()
            cat_long.columns = ["persona_name", "category", "share"]
            cat_long["category"] = cat_long["category"].str.replace("category_share_", "")
            fig_cat = px.bar(
                cat_long, x="persona_name", y="share", color="category", barmode="stack",
                color_discrete_sequence=["#000000", "#404040", "#808080", "#c0c0c0"],
                labels=dict(share="Share", persona_name=""),
            )
            fig_cat.update_layout(**plotly_dior_layout())
            fig_cat.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_cat, use_container_width=True, config=PLOTLY_CONFIG)

    # --- Scatter: recency vs monetary (interactive) ---
    st.markdown('<p class="section-head">Recency vs total spend</p>', unsafe_allow_html=True)
    scatter_df = df[["persona_name", "recency_days", "monetary_total", "frequency_orders"]].copy()
    fig_scatter = px.scatter(
        scatter_df, x="recency_days", y="monetary_total", color="persona_name",
        size="frequency_orders", hover_data=["frequency_orders"],
        color_discrete_sequence=["#000000", "#333333", "#666666", "#999999", "#cccccc"],
        labels=dict(recency_days="Recency (days)", monetary_total="Total spend (€)", persona_name="Persona"),
    )
    fig_scatter.update_layout(**plotly_dior_layout())
    fig_scatter.update_traces(marker=dict(line=dict(width=1, color="#fff")), selector=dict(mode="markers"))
    st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)

    # --- Customer explorer ---
    st.markdown('<p class="section-head">Customer explorer</p>', unsafe_allow_html=True)
    personas = sorted(df["persona_name"].unique().tolist())
    chosen = st.selectbox("Filter by persona", options=["All"] + personas, label_visibility="collapsed")
    subset = df[df["persona_name"] == chosen] if chosen != "All" else df
    n_show = min(50, len(subset))
    cols_display = [c for c in ["customer_id", "persona_name", "monetary_total", "frequency_orders", "recency_days", "avg_basket_value", "boutique_rate"] if c in subset.columns]
    st.dataframe(subset[cols_display].head(n_show), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to {n_show} of {len(subset)} customers.")


if __name__ == "__main__":
    main()
