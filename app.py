"""
Amazon Product Analytics Dashboard
====================================
A futuristic, glassmorphic Streamlit dashboard built directly on top of the
cleaning pipeline and EDA performed in `amazon_eda.ipynb`.

Data lineage (mirrors the notebook exactly):
    1. Raw file: data/amazon_product.csv
    2. Null-handling (cell 10, 12, 14, 16, 18, 20, 22 in the notebook):
       - product_original_price -> filled with product_price
       - product_star_rating    -> filled with median
       - sales_volume           -> filled with 'unknown'
       - delivery               -> filled with mode
       - product_availability   -> filled with 'unknown'
       - unit_price             -> filled with 'unknown'
       - unit_count             -> filled with median
    3. Saved as data/cleaned_amazon_products.csv (cell 26) -- NOTE: at this
       point product_price / product_original_price / product_minimum_offer_price
       are still strings like "$99.99" because the $-stripping happens later
       in the notebook (cells 36, 43, 59).
    4. Price columns stripped of "$" and cast to float (cells 36, 43, 59).
    5. Rows without real sales-volume data ("unknown") dropped by keeping only
       rows where sales_volume contains "bought" (cells 40 / 55).

This app re-applies steps 4-5 on top of the cleaned CSV so the dashboard uses
the exact same final dataframe the notebook's bivariate-analysis / heatmap
cells operated on.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# PAGE CONFIG & GLOBAL CONSTANTS
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Amazon Product Analytics Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette used consistently across every chart + the CSS theme
COLOR_CYAN = "#22d3ee"
COLOR_PURPLE = "#a78bfa"
COLOR_EMERALD = "#34d399"
COLOR_AMBER = "#fbbf24"
COLOR_ROSE = "#fb7185"
ACCENT_SEQUENCE = [COLOR_CYAN, COLOR_PURPLE, COLOR_EMERALD, COLOR_AMBER, COLOR_ROSE]

PLOTLY_TEMPLATE = "plotly_dark"

RAW_PATH_CANDIDATES = ["data/amazon_product.csv", "amazon_product.csv"]
CLEANED_PATH_CANDIDATES = [
    "data/cleaned_amazon_products.csv",
    "cleaned_amazon_products.csv",
]


# --------------------------------------------------------------------------- #
# STYLING — dark, glassmorphic, futuristic AI aesthetic
# --------------------------------------------------------------------------- #

def inject_custom_css() -> None:
    """Inject the dark glassmorphism theme (cards, header, sidebar, tables)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(34, 211, 238, 0.10), transparent 45%),
                radial-gradient(circle at 85% 10%, rgba(167, 139, 250, 0.12), transparent 45%),
                radial-gradient(circle at 50% 100%, rgba(52, 211, 153, 0.08), transparent 55%),
                #05070d;
            color: #e6edf3;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(13, 17, 28, 0.98) 0%, rgba(8, 11, 20, 0.98) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        section[data-testid="stSidebar"] * { color: #cbd5e1; }

        /* ---- Header ---- */
        .dash-header {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            margin-bottom: 1.6rem;
            background: linear-gradient(120deg, rgba(34, 211, 238, 0.10), rgba(167, 139, 250, 0.10) 55%, rgba(52, 211, 153, 0.10));
            border: 1px solid rgba(148, 163, 184, 0.15);
            box-shadow: 0 0 40px rgba(34, 211, 238, 0.06), inset 0 1px 0 rgba(255,255,255,0.04);
            position: relative;
            overflow: hidden;
        }
        .dash-header::before {
            content: "";
            position: absolute; top: -60%; left: -10%;
            width: 50%; height: 220%;
            background: linear-gradient(120deg, transparent, rgba(34,211,238,0.08), transparent);
            transform: rotate(20deg);
            animation: sheen 8s ease-in-out infinite;
        }
        @keyframes sheen {
            0%   { transform: translateX(-40%) rotate(20deg); }
            50%  { transform: translateX(120%) rotate(20deg); }
            100% { transform: translateX(-40%) rotate(20deg); }
        }
        .dash-title {
            font-size: 2.35rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #67e8f9, #c4b5fd 45%, #6ee7b7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }
        .dash-subtitle {
            margin-top: 0.5rem;
            font-size: 1.02rem;
            color: #94a3b8;
            max-width: 780px;
            line-height: 1.5;
        }
        .dash-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.04em;
            background: rgba(52, 211, 153, 0.12);
            border: 1px solid rgba(52, 211, 153, 0.35);
            color: #6ee7b7;
            margin-right: 0.5rem;
        }

        /* ---- Glass cards (KPIs) ---- */
        .glass-card {
            background: rgba(255, 255, 255, 0.035);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 18px;
            padding: 1.3rem 1.4rem;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            height: 100%;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(34, 211, 238, 0.4);
            box-shadow: 0 10px 30px rgba(34, 211, 238, 0.10);
        }
        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #f1f5f9;
            line-height: 1.15;
        }
        .kpi-accent { height: 3px; width: 42px; border-radius: 3px; margin-top: 0.7rem; }

        /* ---- Section headers ---- */
        .section-header {
            display: flex; align-items: center; gap: 0.6rem;
            margin: 1.9rem 0 0.9rem 0;
        }
        .section-header h3 {
            margin: 0; font-size: 1.25rem; font-weight: 600; color: #e6edf3;
        }
        .section-dot {
            width: 9px; height: 9px; border-radius: 50%;
            background: linear-gradient(90deg, #22d3ee, #a78bfa);
            box-shadow: 0 0 10px rgba(34, 211, 238, 0.7);
        }

        .chart-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 18px;
            padding: 1rem 1.1rem 0.3rem 1.1rem;
            backdrop-filter: blur(10px);
        }

        .insight-card {
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-left: 3px solid #22d3ee;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.7rem;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }

        div[data-testid="stMetricValue"] { color: #f1f5f9; }

        hr { border-color: rgba(148, 163, 184, 0.12); }

        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: rgba(34, 211, 238, 0.35); border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, accent: str) -> str:
    """Return HTML for a single glassmorphic KPI card."""
    return f"""
    <div class="glass-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-accent" style="background:{accent};"></div>
    </div>
    """


def section_header(title: str) -> None:
    """Render a consistent glowing section header."""
    st.markdown(
        f"""<div class="section-header"><span class="section-dot"></span><h3>{title}</h3></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# DATA LOADING & CLEANING  (mirrors the notebook pipeline exactly)
# --------------------------------------------------------------------------- #

def _clean_price_column(series: pd.Series) -> pd.Series:
    """Strip a leading '$' and cast to float — identical to notebook cells 36/43/59.

    Handles the column whether it is already numeric (edge case) or a
    '$xx.xx' string, and coerces unparseable values to NaN rather than
    raising, since raw scraped data can contain stray formatting issues.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.strip(),
        errors="coerce",
    )


def _apply_notebook_null_handling(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's null-handling cells (10, 12, 14, 16, 18, 20, 22).

    Only applied when a column is present and still needs it — this lets the
    same function safely run on either the raw CSV or an already-cleaned CSV.
    """
    if "product_original_price" in df.columns and "product_price" in df.columns:
        df["product_original_price"] = df["product_original_price"].fillna(df["product_price"])
    if "product_star_rating" in df.columns:
        df["product_star_rating"] = df["product_star_rating"].fillna(df["product_star_rating"].median())
    if "sales_volume" in df.columns:
        df["sales_volume"] = df["sales_volume"].fillna("unknown")
    if "delivery" in df.columns and df["delivery"].isnull().any():
        mode_val = df["delivery"].mode()
        if len(mode_val):
            df["delivery"] = df["delivery"].fillna(mode_val[0])
    if "product_availability" in df.columns:
        df["product_availability"] = df["product_availability"].fillna("unknown")
    if "unit_price" in df.columns:
        df["unit_price"] = df["unit_price"].fillna("unknown")
    if "unit_count" in df.columns:
        df["unit_count"] = df["unit_count"].fillna(df["unit_count"].median())
    return df


def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply price type-casting + the 'bought' sales-volume filter (cells 36-60)."""
    for col in ["product_price", "product_original_price", "product_minimum_offer_price"]:
        if col in df.columns:
            df[col] = _clean_price_column(df[col])

    # Cell 40 / 55: keep only rows with real sales-volume signal
    if "sales_volume" in df.columns:
        df = df[df["sales_volume"].astype(str).str.contains("bought", case=False, na=False)]

    # Boolean flag columns used by the sidebar filters
    for col in ["is_best_seller", "is_amazon_choice", "is_prime", "climate_pledge_friendly", "has_variations"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df.reset_index(drop=True)


def _first_existing_path(candidates: list[str]) -> str | None:
    import os
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    """Load the Amazon dataset, applying the notebook's cleaning pipeline.

    Preference order:
        1. A user-uploaded CSV (sidebar fallback)
        2. The pre-cleaned CSV the notebook exports (data/cleaned_amazon_products.csv)
        3. The raw CSV (data/amazon_product.csv), fully re-cleaned in-app

    Returns an empty DataFrame with the expected schema if nothing is found,
    so the rest of the app can render gracefully with a clear warning instead
    of crashing.
    """
    if uploaded_bytes is not None:
        df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes))
        df = _apply_notebook_null_handling(df)
        return _finalize_dataframe(df)

    cleaned_path = _first_existing_path(CLEANED_PATH_CANDIDATES)
    if cleaned_path:
        df = pd.read_csv(cleaned_path)
        df = _apply_notebook_null_handling(df)  # no-op if already clean
        return _finalize_dataframe(df)

    raw_path = _first_existing_path(RAW_PATH_CANDIDATES)
    if raw_path:
        df = pd.read_csv(raw_path)
        df = _apply_notebook_null_handling(df)
        return _finalize_dataframe(df)

    return pd.DataFrame()


# --------------------------------------------------------------------------- #
# SIDEBAR FILTERS
# --------------------------------------------------------------------------- #

def render_sidebar_filters(df: pd.DataFrame) -> dict:
    """Render sidebar filter widgets and return the selected filter values."""
    st.sidebar.markdown("## 🛰️ Filters")
    st.sidebar.markdown("Refine the product universe in real time.")
    st.sidebar.markdown("---")

    filters = {}

    # --- Price range ---
    if "product_price" in df.columns and df["product_price"].notna().any():
        p_min = float(np.floor(df["product_price"].min()))
        p_max = float(np.ceil(df["product_price"].max()))
        if p_min == p_max:
            p_max += 1.0
        filters["price_range"] = st.sidebar.slider(
            "💲 Price Range",
            min_value=p_min,
            max_value=p_max,
            value=(p_min, p_max),
            step=1.0,
        )
    else:
        filters["price_range"] = None

    # --- Star rating ---
    if "product_star_rating" in df.columns and df["product_star_rating"].notna().any():
        r_min = float(np.floor(df["product_star_rating"].min()))
        r_max = float(np.ceil(df["product_star_rating"].max()))
        filters["rating_range"] = st.sidebar.slider(
            "⭐ Star Rating",
            min_value=r_min,
            max_value=r_max,
            value=(r_min, r_max),
            step=0.1,
        )
    else:
        filters["rating_range"] = None

    st.sidebar.markdown("---")

    # --- Boolean toggles ---
    filters["best_seller"] = st.sidebar.selectbox(
        "🏆 Best Seller", options=["All", "Best Sellers Only", "Exclude Best Sellers"]
    )
    filters["prime"] = st.sidebar.selectbox(
        "📦 Prime Products", options=["All", "Prime Only", "Non-Prime Only"]
    )
    filters["amazon_choice"] = st.sidebar.selectbox(
        "✅ Amazon Choice", options=["All", "Amazon's Choice Only", "Exclude Amazon's Choice"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<span style='font-size:0.8rem;color:#64748b;'>Dashboard built on the "
        "cleaning pipeline & EDA from <code>amazon_eda.ipynb</code>.</span>",
        unsafe_allow_html=True,
    )

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply the sidebar filter selections to the dataframe."""
    filtered = df.copy()

    if filters.get("price_range") and "product_price" in filtered.columns:
        lo, hi = filters["price_range"]
        filtered = filtered[filtered["product_price"].between(lo, hi)]

    if filters.get("rating_range") and "product_star_rating" in filtered.columns:
        lo, hi = filters["rating_range"]
        filtered = filtered[filtered["product_star_rating"].between(lo, hi)]

    if "is_best_seller" in filtered.columns:
        if filters.get("best_seller") == "Best Sellers Only":
            filtered = filtered[filtered["is_best_seller"] == True]  # noqa: E712
        elif filters.get("best_seller") == "Exclude Best Sellers":
            filtered = filtered[filtered["is_best_seller"] == False]  # noqa: E712

    if "is_prime" in filtered.columns:
        if filters.get("prime") == "Prime Only":
            filtered = filtered[filtered["is_prime"] == True]  # noqa: E712
        elif filters.get("prime") == "Non-Prime Only":
            filtered = filtered[filtered["is_prime"] == False]  # noqa: E712

    if "is_amazon_choice" in filtered.columns:
        if filters.get("amazon_choice") == "Amazon's Choice Only":
            filtered = filtered[filtered["is_amazon_choice"] == True]  # noqa: E712
        elif filters.get("amazon_choice") == "Exclude Amazon's Choice":
            filtered = filtered[filtered["is_amazon_choice"] == False]  # noqa: E712

    return filtered


# --------------------------------------------------------------------------- #
# KPI METRICS
# --------------------------------------------------------------------------- #

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the four headline KPI values, guarding against empty frames."""
    if df.empty:
        return {"total_products": 0, "avg_price": 0.0, "avg_rating": 0.0, "total_ratings": 0}
    return {
        "total_products": int(len(df)),
        "avg_price": float(df["product_price"].mean()) if "product_price" in df else 0.0,
        "avg_rating": float(df["product_star_rating"].mean()) if "product_star_rating" in df else 0.0,
        "total_ratings": int(df["product_num_ratings"].sum()) if "product_num_ratings" in df else 0,
    }


def render_kpi_row(kpis: dict) -> None:
    """Render the four KPI glass cards in a responsive row."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Products", f"{kpis['total_products']:,}", COLOR_CYAN), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Avg. Product Price", f"${kpis['avg_price']:,.2f}", COLOR_PURPLE), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg. Star Rating", f"{kpis['avg_rating']:.2f} ★", COLOR_EMERALD), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Total Ratings Collected", f"{kpis['total_ratings']:,}", COLOR_AMBER), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# CHART BUILDERS  (each mirrors a specific notebook cell)
# --------------------------------------------------------------------------- #

def _style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the shared dark/glass styling to every Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=10, r=10, t=45, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.12)")
    return fig


def chart_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Product Price Distribution — mirrors notebook cell 38 (histplot)."""
    fig = px.histogram(df, x="product_price", nbins=30, color_discrete_sequence=[COLOR_CYAN])
    fig.update_layout(title="Distribution of Product Price", xaxis_title="Product Price ($)", yaxis_title="Count")
    return _style_fig(fig)


def chart_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Star Rating Distribution — mirrors notebook cell 28 (histplot)."""
    fig = px.histogram(df, x="product_star_rating", nbins=10, color_discrete_sequence=[COLOR_EMERALD])
    fig.update_layout(title="Distribution of Product Star Ratings", xaxis_title="Star Rating", yaxis_title="Count")
    return _style_fig(fig)


def chart_price_vs_original(df: pd.DataFrame) -> go.Figure:
    """Price vs Original Price — mirrors notebook cell 60 (regplot)."""
    fig = px.scatter(
        df, x="product_original_price", y="product_price",
        opacity=0.55, color_discrete_sequence=[COLOR_PURPLE],
        trendline="ols",
    )
    fig.update_traces(marker=dict(size=6))
    for trace in fig.data:
        if trace.mode == "lines":
            trace.line.color = COLOR_ROSE
            trace.line.width = 2.5
    fig.update_layout(title="Product Price vs Original Price", xaxis_title="Original Price ($)", yaxis_title="Current Price ($)")
    return _style_fig(fig)


def chart_price_vs_num_ratings(df: pd.DataFrame) -> go.Figure:
    """Price vs Number of Ratings — mirrors notebook cell 50 (scatter, log-y)."""
    fig = px.scatter(
        df, x="product_price", y="product_num_ratings",
        opacity=0.55, color_discrete_sequence=[COLOR_AMBER],
    )
    fig.update_yaxes(type="log")
    fig.update_layout(title="Product Price vs Number of Ratings (log scale)", xaxis_title="Product Price ($)", yaxis_title="Number of Ratings (log)")
    return _style_fig(fig)


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation Heatmap — mirrors notebook cell 62, computed on the final
    (post price-conversion) numeric columns so it reflects the actual numeric
    fields available at this point in the pipeline."""
    numeric_cols = [
        c for c in [
            "product_price", "product_original_price", "product_star_rating",
            "product_num_ratings", "product_num_offers",
            "product_minimum_offer_price", "unit_count",
        ] if c in df.columns
    ]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=[COLOR_PURPLE, "#0f172a", COLOR_CYAN],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(title="Correlation Matrix — Numerical Variables")
    return _style_fig(fig, height=440)


def chart_num_ratings_distribution(df: pd.DataFrame) -> go.Figure:
    """Num Ratings Distribution — mirrors notebook cell 30 (log1p histplot)."""
    fig = px.histogram(df, x=np.log1p(df["product_num_ratings"]), nbins=20, color_discrete_sequence=[COLOR_EMERALD])
    fig.update_layout(title="Distribution of Number of Ratings", xaxis_title="log1p(Num Ratings)", yaxis_title="Count")
    return _style_fig(fig)


def chart_num_offers_distribution(df: pd.DataFrame) -> go.Figure:
    """Num Offers Distribution — mirrors notebook cell 32 (log1p histplot)."""
    fig = px.histogram(df, x=np.log1p(df["product_num_offers"]), nbins=10, color_discrete_sequence=[COLOR_CYAN])
    fig.update_layout(title="Distribution of Number of Offers", xaxis_title="log1p(Num Offers)", yaxis_title="Count")
    return _style_fig(fig)


def chart_unit_count_distribution(df: pd.DataFrame) -> go.Figure:
    """Unit Count Distribution — mirrors notebook cell 34 (log1p histplot)."""
    fig = px.histogram(df, x=np.log1p(df["unit_count"]), nbins=10, color_discrete_sequence=[COLOR_PURPLE])
    fig.update_layout(title="Distribution of Unit Count (log scale)", xaxis_title="log1p(Unit Count)", yaxis_title="Count")
    return _style_fig(fig)


def chart_sales_volume_countplot(df: pd.DataFrame) -> go.Figure:
    """Sales Volume distribution — mirrors notebook cell 41 (countplot)."""
    counts = df["sales_volume"].value_counts().reset_index()
    counts.columns = ["sales_volume", "count"]
    fig = px.bar(
        counts.sort_values("count"), x="count", y="sales_volume", orientation="h",
        color="count", color_continuous_scale=[COLOR_CYAN, COLOR_PURPLE],
    )
    fig.update_layout(title="Distribution of Sales Volume", xaxis_title="Count", yaxis_title="Sales Volume")
    return _style_fig(fig, height=440)


def chart_min_offer_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Minimum Offer Price Distribution — mirrors notebook cell 45 (log1p histplot)."""
    fig = px.histogram(df, x=np.log1p(df["product_minimum_offer_price"]), nbins=10, color_discrete_sequence=[COLOR_CYAN])
    fig.update_layout(title="Distribution of Minimum Offer Price", xaxis_title="log1p(Minimum Offer Price)", yaxis_title="Count")
    return _style_fig(fig)


def chart_price_vs_star_rating(df: pd.DataFrame) -> go.Figure:
    """Price vs Star Rating — mirrors notebook cell 48 (scatter)."""
    fig = px.scatter(df, x="product_price", y="product_star_rating", opacity=0.55, color_discrete_sequence=[COLOR_EMERALD])
    fig.update_layout(title="Product Price vs Star Rating", xaxis_title="Product Price ($)", yaxis_title="Star Rating")
    return _style_fig(fig)


def chart_price_vs_num_offers(df: pd.DataFrame) -> go.Figure:
    """Price vs Number of Offers — mirrors notebook cell 52 (scatter)."""
    fig = px.scatter(df, x="product_price", y="product_num_offers", opacity=0.4, color_discrete_sequence=[COLOR_EMERALD])
    fig.update_layout(title="Product Price vs Number of Offers", xaxis_title="Product Price ($)", yaxis_title="Number of Offers")
    return _style_fig(fig)


def chart_rating_vs_num_ratings(df: pd.DataFrame) -> go.Figure:
    """Star Rating vs Num Ratings — mirrors notebook cell 54 (scatter, log-y)."""
    fig = px.scatter(df, x="product_star_rating", y="product_num_ratings", opacity=0.55, color_discrete_sequence=[COLOR_PURPLE])
    fig.update_yaxes(type="log")
    fig.update_layout(title="Star Rating vs Number of Ratings (log scale)", xaxis_title="Star Rating", yaxis_title="Number of Ratings (log)")
    return _style_fig(fig)


def chart_rating_by_sales_volume(df: pd.DataFrame) -> go.Figure:
    """Star Rating by Sales Volume — mirrors notebook cell 57 (boxplot)."""
    order = df["sales_volume"].value_counts().index.tolist()
    fig = px.box(
        df, x="product_star_rating", y="sales_volume", category_orders={"sales_volume": order},
        color_discrete_sequence=[COLOR_AMBER],
    )
    fig.update_layout(title="Product Star Rating by Sales Volume", xaxis_title="Star Rating", yaxis_title="Sales Volume")
    return _style_fig(fig, height=460)


# --------------------------------------------------------------------------- #
# INSIGHTS  (computed live from the current filtered data, not hardcoded)
# --------------------------------------------------------------------------- #

def render_insights(df: pd.DataFrame, full_df: pd.DataFrame) -> None:
    """Summarize key EDA findings, computed dynamically so figures always
    reflect the actual current dataset rather than stale, hardcoded numbers.
    """
    if df.empty:
        st.info("No products match the current filters — adjust the filters to see insights.")
        return

    insights = []

    # Price vs original price relationship
    if {"product_price", "product_original_price"}.issubset(df.columns):
        corr_price_orig = df["product_price"].corr(df["product_original_price"])
        discounted = (df["product_price"] < df["product_original_price"]).mean() * 100
        insights.append(
            f"<b>Price vs. Original Price:</b> correlation of <b>{corr_price_orig:.2f}</b> — "
            f"current price tracks list price closely. About <b>{discounted:.1f}%</b> of products "
            f"in the current view are priced below their original price (i.e. discounted)."
        )

    # Price vs rating
    if {"product_price", "product_star_rating"}.issubset(df.columns):
        corr_price_rating = df["product_price"].corr(df["product_star_rating"])
        insights.append(
            f"<b>Price vs. Star Rating:</b> correlation of <b>{corr_price_rating:.2f}</b> — "
            f"{'a weak' if abs(corr_price_rating) < 0.2 else 'a noticeable'} relationship, "
            f"suggesting price alone is a poor predictor of how a product is rated."
        )

    # Price vs number of ratings (popularity proxy)
    if {"product_price", "product_num_ratings"}.issubset(df.columns):
        corr_price_numrat = df["product_price"].corr(df["product_num_ratings"])
        insights.append(
            f"<b>Price vs. Number of Ratings:</b> correlation of <b>{corr_price_numrat:.2f}</b> — "
            f"{'cheaper products tend to accumulate more ratings' if corr_price_numrat < -0.1 else 'ratings volume is roughly independent of price'}."
        )

    # Rating distribution skew
    if "product_star_rating" in df.columns:
        median_rating = df["product_star_rating"].median()
        pct_above_4 = (df["product_star_rating"] >= 4.0).mean() * 100
        insights.append(
            f"<b>Rating Skew:</b> median star rating is <b>{median_rating:.1f}</b>, and "
            f"<b>{pct_above_4:.1f}%</b> of products hold a rating of 4.0 or higher — ratings "
            f"are left-skewed, as is typical for Amazon listings."
        )

    # Best seller / Prime / Amazon Choice composition
    comp_bits = []
    if "is_best_seller" in df.columns:
        comp_bits.append(f"{df['is_best_seller'].mean()*100:.1f}% Best Sellers")
    if "is_prime" in df.columns:
        comp_bits.append(f"{df['is_prime'].mean()*100:.1f}% Prime-eligible")
    if "is_amazon_choice" in df.columns:
        comp_bits.append(f"{df['is_amazon_choice'].mean()*100:.1f}% Amazon's Choice")
    if comp_bits:
        insights.append(f"<b>Catalog Composition:</b> of the current selection, " + ", ".join(comp_bits) + ".")

    # Sales volume signal
    if "sales_volume" in df.columns and not df["sales_volume"].empty:
        top_bucket = df["sales_volume"].value_counts().idxmax()
        insights.append(
            f"<b>Sales Volume:</b> the most common sales-volume bucket in the current view is "
            f"\"<b>{top_bucket}</b>\" (rows lacking real sales-volume data were dropped during "
            f"cleaning, matching the notebook's filtering step)."
        )

    for text in insights:
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

    st.caption(
        f"Computed on {len(df):,} of {len(full_df):,} total cleaned records currently matching the sidebar filters."
    )


# --------------------------------------------------------------------------- #
# MAIN APP
# --------------------------------------------------------------------------- #

def main() -> None:
    inject_custom_css()

    # ---- Header ----
    st.markdown(
        """
        <div class="dash-header">
            <span class="dash-badge">● LIVE ANALYTICS</span>
            <span class="dash-badge">EDA-DRIVEN</span>
            <h1 class="dash-title">Amazon Product Analytics Dashboard</h1>
            <p class="dash-subtitle">
                An interactive, AI-inspired command center for exploring Amazon product listings —
                pricing, ratings, discounts, and sales-volume signal — built directly on top of the
                cleaning pipeline and exploratory analysis performed in <code>amazon_eda.ipynb</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()

    if df.empty:
        st.warning(
            "No dataset found at `data/cleaned_amazon_products.csv` or `data/amazon_product.csv`. "
            "Upload the CSV to continue."
        )
        uploaded = st.file_uploader("Upload amazon_product.csv or cleaned_amazon_products.csv", type=["csv"])
        if uploaded is not None:
            df = load_data(uploaded_bytes=uploaded.getvalue())
        if df.empty:
            st.stop()

    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    # ---- KPI Cards ----
    section_header("Key Performance Indicators")
    render_kpi_row(compute_kpis(filtered_df))

    if filtered_df.empty:
        st.info("No products match the selected filters. Try widening the price or rating range.")
        st.stop()

    # ---- Visualizations: primary required set ----
    section_header("Core Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_price_distribution(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_rating_distribution(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_price_vs_original(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_price_vs_num_ratings(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(chart_correlation_heatmap(filtered_df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Additional visualizations from the notebook ----
    section_header("Additional EDA Visualizations")
    tabs = st.tabs([
        "Univariate", "Bivariate", "Sales Volume",
    ])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_num_ratings_distribution(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_num_offers_distribution(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_unit_count_distribution(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_min_offer_price_distribution(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_price_vs_star_rating(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(chart_price_vs_num_offers(filtered_df), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_rating_vs_num_ratings(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_sales_volume_countplot(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(chart_rating_by_sales_volume(filtered_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Interactive Data Table ----
    section_header("Product Explorer")
    search_col, _ = st.columns([2, 3])
    with search_col:
        search_term = st.text_input("🔍 Search by product title", "")

    table_df = filtered_df.copy()
    if search_term:
        table_df = table_df[table_df["product_title"].str.contains(search_term, case=False, na=False)]

    display_cols = [
        c for c in [
            "product_title", "product_price", "product_original_price", "product_star_rating",
            "product_num_ratings", "product_num_offers", "is_best_seller", "is_prime",
            "is_amazon_choice", "sales_volume", "delivery",
        ] if c in table_df.columns
    ]
    st.dataframe(
        table_df[display_cols].sort_values("product_price"),
        use_container_width=True,
        height=420,
    )
    st.caption(f"Showing {len(table_df):,} products (sortable by clicking column headers).")

    # ---- Insights ----
    section_header("Key Insights")
    render_insights(filtered_df, df)

    st.markdown(
        "<p style='text-align:center;color:#475569;font-size:0.8rem;margin-top:2rem;'>"
        "Amazon Product Analytics Dashboard · Built with Streamlit · Data cleaned & explored in amazon_eda.ipynb"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()