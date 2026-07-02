"""
BathExpertz - Senior Management Dashboard
Grain: 1 row = 1 bathroom (Project Child Code). Multiple bathrooms can belong
to the same customer (Project Parent Code).

No pivot tables used anywhere - all aggregation via pandas groupby/agg.
"""

import hashlib

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="BathExpertz - Management Dashboard", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 17px !important; }
[data-testid="stMetricValue"] { font-size: 2.1rem !important; }
[data-testid="stMetricLabel"] { font-size: 1.05rem !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab"] { font-size: 1.15rem !important; font-weight: 600 !important; padding: 10px 18px !important; }
h1 { font-size: 2.3rem !important; }
h2 { font-size: 1.7rem !important; }
h3 { font-size: 1.4rem !important; }
p, li, .stMarkdown, label { font-size: 1.05rem !important; line-height: 1.55 !important; }
[data-testid="stDataFrame"] { font-size: 1.02rem !important; }
[data-testid="stCaptionContainer"] { font-size: 1rem !important; }

.kpi-card {
    border-radius: 14px;
    padding: 18px 20px 14px 20px;
    color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-label { font-size: 0.95rem; font-weight: 600; opacity: 0.92; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 800; line-height: 1.15; }
.kpi-sub   { font-size: 0.82rem; opacity: 0.85; margin-top: 2px; }
.insight-box {
    background: #F0F7FF;
    border-left: 5px solid #2E86AB;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 1.02rem;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHART THEME
# ---------------------------------------------------------------------------
COLOR_SEQ = ["#2E86AB", "#F26419", "#33A02C", "#8E44AD", "#F6AE2D", "#E63946",
             "#118AB2", "#06D6A0", "#EF476F", "#FFB703", "#073B4C"]
ACCENT = "#2E86AB"
KPI_COLORS = ["#2E86AB", "#33A02C", "#F26419", "#8E44AD", "#118AB2", "#E63946", "#06D6A0", "#F6AE2D"]


def style_fig(fig, height=420, showlegend=None, title=None):
    fig.update_layout(
        font=dict(size=15, family="Arial"),
        title_font_size=19,
        legend=dict(font=dict(size=13)),
        height=height,
        margin=dict(t=50, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title is not None:
        fig.update_layout(title=title)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    fig.update_xaxes(title_font=dict(size=15), tickfont=dict(size=13))
    fig.update_yaxes(title_font=dict(size=15), tickfont=dict(size=13))
    return fig


def kpi_card(label, value, sub=None, color=ACCENT):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card" style="background:{color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def kpi_row(items):
    """items: list of (label, value, sub, color) tuples -> renders as a row of colored boxes."""
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        with col:
            st.markdown(kpi_card(label, value, sub, color), unsafe_allow_html=True)


def counts_table(series, label_col, count_col="Count"):
    """Safe replacement for value_counts().reset_index(name=...) — avoids the
    unnamed/misaligned-column issue that fed 'undefined' labels into charts."""
    vc = series.dropna().value_counts()
    out = pd.DataFrame({label_col: vc.index.astype(str), count_col: vc.values})
    return out.reset_index(drop=True)


def safe_pct(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator * 100


def fmt_pct(x):
    return f"{x:.1f}%" if pd.notna(x) else "0.0%"


DATA_PATH = "cleaned_data.csv"

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    date_cols = ["Booking Date", "60% Date", "20% date", "30% date", "50% date",
                 "Handover Date", "Remaining Amount date",
                 "Booking Month", "60% Month", "Handover Month"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["Months to Design completed from booking", "Months to Execution completed from Design"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_raw = load_data()

# ---------------------------------------------------------------------------
# CUSTOMER-LEVEL "SMART SUM" — fixes the repeated-quotation-value data issue
# ---------------------------------------------------------------------------
def customer_level_value(frame, value_col):
    def _agg(s):
        s = s.dropna()
        if len(s) == 0:
            return 0.0
        return s.iloc[0] if s.nunique() == 1 else s.sum()
    return frame.groupby("Project Parent Code")[value_col].apply(_agg)

def smart_total(frame, value_col):
    return customer_level_value(frame, value_col).sum()

# ---------------------------------------------------------------------------
# APPROX ZONE CENTROIDS (Delhi-NCR) — used for the geography map.
# Source data has no lat/lon, so these are zone-centroid approximations,
# with a small deterministic jitter per locality so bubbles don't fully overlap.
# ---------------------------------------------------------------------------
ZONE_COORDS = {
    "Gurgaon": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
    "Greater Noida": (28.4744, 77.5040),
    "Faridabad": (28.4089, 77.3178),
    "Ghaziabad": (28.6692, 77.4538),
    "South Delhi": (28.5245, 77.2066),
    "North Delhi": (28.7041, 77.1910),
    "West Delhi": (28.6692, 77.0910),
    "Central Delhi": (28.6519, 77.2315),
    "East Delhi": (28.6279, 77.2773),
}
DEFAULT_COORD = (28.6139, 77.2090)  # Delhi center, fallback for "Other"/unmapped


def jitter(seed_text, scale=0.025):
    h = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)
    dx = ((h % 1000) / 1000 - 0.5) * 2 * scale
    dy = (((h // 1000) % 1000) / 1000 - 0.5) * 2 * scale
    return dx, dy


def locality_coords(zone, locality):
    base = ZONE_COORDS.get(zone, DEFAULT_COORD)
    dx, dy = jitter(f"{zone}|{locality}")
    return base[0] + dy, base[1] + dx

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

min_d, max_d = df_raw["Booking Date"].min(), df_raw["Booking Date"].max()
date_range = st.sidebar.date_input("Booking Date range", value=(min_d, max_d),
                                    min_value=min_d, max_value=max_d)

zones = st.sidebar.multiselect("Zone", sorted(df_raw["Zone"].dropna().unique()))
teams = st.sidebar.multiselect("Sales Team", sorted(df_raw["Sales team"].dropna().unique()))
statuses = st.sidebar.multiselect("Final Project Status", sorted(df_raw["Final Project status"].dropna().unique()))

df = df_raw.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["Booking Date"] >= pd.Timestamp(date_range[0])) &
            (df["Booking Date"] <= pd.Timestamp(date_range[1]))]
if zones:
    df = df[df["Zone"].isin(zones)]
if teams:
    df = df[df["Sales team"].isin(teams)]
if statuses:
    df = df[df["Final Project status"].isin(statuses)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** bathrooms across **{df['Project Parent Code'].nunique():,}** customers")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🚿 BathExpertz — Management Dashboard")
st.caption("Grain: 1 row = 1 bathroom. A customer (Project Parent Code) may book multiple bathrooms.")

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🧾 Data Snapshot", "📊 Overview", "🔀 Funnel & Conversion", "💰 Revenue & Collections",
     "⏱️ TAT & Aging", "👥 Team Performance", "🗺️ Geographic"]
)

# ===========================================================================
# TAB 0: DATA SNAPSHOT
# ===========================================================================
with tab0:
    st.subheader("What This Data Is")
    st.markdown(
        "Bathroom renovation bookings for **BathExpertz**, a Delhi-NCR based renovation company, "
        "covering the full customer journey: **Sales booking → Design → Execution → Handover**."
    )

    min_date_full = df_raw["Booking Date"].min()
    max_date_full = df_raw["Booking Date"].max()
    n_cust_full = df_raw["Project Parent Code"].nunique()

    kpi_row([
        ("Data Covers", f"{min_date_full.date()} → {max_date_full.date()}", None, KPI_COLORS[0]),
        ("Bathrooms (rows)", f"{len(df_raw):,}", None, KPI_COLORS[1]),
        ("Unique Customers", f"{n_cust_full:,}", None, KPI_COLORS[2]),
        ("Avg Bathrooms / Customer", f"{len(df_raw)/n_cust_full:.2f}", None, KPI_COLORS[3]),
    ])
    st.write("")

    last_month_count = df_raw[df_raw["Booking Date"].dt.to_period("M") == max_date_full.to_period("M")].shape[0]
    st.warning(
        f"⚠️ The most recent month ({max_date_full.strftime('%b %Y')}) has only "
        f"{last_month_count} bookings vs. 70-150 in prior months — this is a **partial month** "
        f"(data pulled mid-month), not a real demand drop. Filter it out for trend comparisons."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Grain of the data**")
        st.markdown(
            "- 1 row = **1 bathroom** (`Project Child Code`)\n"
            "- A customer (`Project Parent Code`) can book multiple bathrooms — "
            "up to 10 in this dataset\n"
            "- Customer-level metrics use `nunique()` on Parent Code; "
            "bathroom-level metrics use row count"
        )
        st.markdown("**Coverage**")
        st.markdown(
            f"- **{df_raw['Zone'].nunique()} zones**, **{df_raw['Locality'].nunique()} localities** "
            f"(Gurgaon, South/North/West/Central/East Delhi, Noida, Greater Noida, Faridabad, Ghaziabad, Other)\n"
            f"- **{df_raw['Sales team'].nunique()} sales reps**"
        )
    with col_b:
        st.markdown("**Final Project Status — full breakdown**")
        snap_status = counts_table(df_raw["Final Project status"], "Status", "Bathrooms")
        st.dataframe(snap_status, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Known data quality fixes applied during cleaning**")
    st.markdown(
        "- Duplicate status labels folded in: `Lost In Design` → `Lost at Design`, "
        "`Lost At Execution` → `Lost at Execution`\n"
        "- Project code casing standardized (`Bx-2354` → `BX-2354`)\n"
        "- `Booking/60%/Handover Month` recomputed from source dates (originals were formula artifacts)\n"
        "- Revenue figures de-duplicated at customer level (see Overview tab note) — "
        "**no rows or columns were dropped**"
    )

# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab1:
    st.subheader("Business Snapshot")

    n_bathrooms = len(df)
    n_customers = df["Project Parent Code"].nunique()
    avg_bathrooms_per_cust = n_bathrooms / n_customers if n_customers else 0
    total_quotation = smart_total(df, "Quotation value")
    total_revenue = smart_total(df, "Total revenue collected")
    collection_pct = safe_pct(total_revenue, total_quotation)
    handover_ct = int((df["Final Project status"] == "Handover Completed").sum())
    handover_pct = safe_pct(handover_ct, n_bathrooms)
    lost_mask = df["Final Project status"].isin(["Lost at Design", "Lost at Execution", "Cancelled at sales"])
    lost_ct = int(lost_mask.sum())
    lost_pct = safe_pct(lost_ct, n_bathrooms)

    kpi_row([
        ("Bathrooms Booked", f"{n_bathrooms:,}", None, KPI_COLORS[0]),
        ("Unique Customers", f"{n_customers:,}", None, KPI_COLORS[1]),
        ("Avg Bathrooms / Customer", f"{avg_bathrooms_per_cust:.2f}", None, KPI_COLORS[2]),
        ("Handover Completed", fmt_pct(handover_pct), f"{handover_ct:,} of {n_bathrooms:,}", KPI_COLORS[3]),
    ])
    st.write("")
    kpi_row([
        ("Total Quotation Value", f"₹{total_quotation/1e7:.2f} Cr", None, KPI_COLORS[4]),
        ("Total Revenue Collected", f"₹{total_revenue/1e7:.2f} Cr", None, KPI_COLORS[5]),
        ("Collection Efficiency", fmt_pct(collection_pct), "Revenue ÷ Quotation value", KPI_COLORS[6]),
        ("Lost / Cancelled", fmt_pct(lost_pct), f"{lost_ct:,} of {n_bathrooms:,}", KPI_COLORS[7]),
    ])

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Bookings Trend by Month**")
        monthly = df.dropna(subset=["Booking Month"]).groupby(
            df["Booking Month"].dt.to_period("M").astype(str)
        ).size().reset_index(name="Bathrooms Booked")
        monthly.columns = ["Month", "Bathrooms Booked"]
        fig = px.bar(monthly, x="Month", y="Bathrooms Booked",
                     color_discrete_sequence=[ACCENT], text="Bathrooms Booked")
        fig.update_traces(textposition="outside", marker_color=ACCENT)
        st.plotly_chart(style_fig(fig, showlegend=False, title="Bathrooms Booked per Month"),
                         use_container_width=True)

    with col_b:
        st.markdown("**Final Project Status Distribution**")
        status_counts = counts_table(df["Final Project status"], "Status", "Count")
        fig = px.bar(status_counts, x="Count", y="Status", orientation="h", color="Status",
                     color_discrete_sequence=COLOR_SEQ, text="Count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_fig(fig, showlegend=False, title="Where Every Bathroom Stands Today"),
                         use_container_width=True)

    st.info(
        "**Data quality note:** 51 of 256 multi-bathroom customers had an identical "
        "Quotation Value repeated across every bathroom row in the source data "
        "(a project-level total, not per-bathroom). All revenue figures on this "
        "dashboard use a customer-level de-duplication rule to avoid double-counting these."
    )

# ===========================================================================
# TAB 2: FUNNEL & CONVERSION
# ===========================================================================
with tab2:
    st.subheader("Sales → Design → Execution Funnel")
    st.markdown(
        '<div class="insight-box">📌 <b>What this tells you:</b> of every 100 bathrooms booked, '
        'how many actually make it to design payment, final payment, and handover — and at which '
        'gate the business loses the most deals.</div>',
        unsafe_allow_html=True,
    )

    booked = len(df)
    reached_design_pay = int(df["60% Date"].notna().sum())
    reached_final_pay = int(df["50% date"].notna().sum())
    handed_over = int((df["Final Project status"] == "Handover Completed").sum())

    funnel_df = pd.DataFrame({
        "Stage": ["Booked", "Reached Design Payment (60%)", "Reached Final Payment (50%)", "Handover Completed"],
        "Bathrooms": [booked, reached_design_pay, reached_final_pay, handed_over]
    })
    # NOTE: no `color=` here — coloring a funnel by stage splits it into single-row
    # traces, which breaks Plotly's "percent of previous" math and prints "undefined%".
    fig = px.funnel(funnel_df, x="Bathrooms", y="Stage")
    fig.update_traces(
        marker_color=[COLOR_SEQ[0], COLOR_SEQ[1], COLOR_SEQ[2], COLOR_SEQ[3]],
        textinfo="value+percent previous",
        texttemplate="%{value:,} (%{percentPrevious})",
    )
    st.plotly_chart(style_fig(fig, height=380, showlegend=False, title="Booking-to-Handover Funnel"),
                     use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Booking → Design Payment", fmt_pct(safe_pct(reached_design_pay, booked)),
              help=f"{reached_design_pay:,} of {booked:,} bathrooms")
    c2.metric("Design Payment → Final Payment", fmt_pct(safe_pct(reached_final_pay, reached_design_pay)),
              help=f"{reached_final_pay:,} of {reached_design_pay:,} bathrooms")
    c3.metric("Final Payment → Handover", fmt_pct(safe_pct(handed_over, reached_final_pay)),
              help=f"{handed_over:,} of {reached_final_pay:,} bathrooms")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Lost / Cancelled Breakdown**")
        lost = df[df["Final Project status"].isin(
            ["Lost at Design", "Lost at Execution", "Cancelled at sales"])]
        lost_counts = counts_table(lost["Final Project status"], "Reason Stage", "Count")
        if len(lost_counts):
            fig = px.pie(lost_counts, names="Reason Stage", values="Count", hole=0.4,
                         color_discrete_sequence=COLOR_SEQ)
            fig.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(style_fig(fig, title="Where Deals Are Lost"), use_container_width=True)
        else:
            st.caption("No lost/cancelled bathrooms in current filter selection.")

    with col_b:
        st.markdown("**On-Hold Design Reasons**")
        holds = counts_table(df["On hold Design Reasons"], "Reason", "Count")
        if len(holds):
            fig = px.bar(holds, x="Count", y="Reason", orientation="h",
                         color_discrete_sequence=[ACCENT], text="Count")
            fig.update_traces(textposition="outside", marker_color=ACCENT)
            st.plotly_chart(style_fig(fig, showlegend=False, title="Why Bathrooms Get Stuck On Hold"),
                             use_container_width=True)
        else:
            st.caption("No on-hold reasons in current filter selection.")

    st.markdown("**Sales / Design / Execution Stage Breakdown**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Sales Booking Stage")
        st.dataframe(counts_table(df["Sales Booking Stage"], "Stage", "Count"),
                     hide_index=True, use_container_width=True)
    with c2:
        st.caption("Design Stage Status")
        st.dataframe(counts_table(df["Design Stage Status"], "Status", "Count"),
                     hide_index=True, use_container_width=True)
    with c3:
        st.caption("Execution Stage Status")
        st.dataframe(counts_table(df["Execution Stage Status"], "Status", "Count"),
                     hide_index=True, use_container_width=True)

# ===========================================================================
# TAB 3: REVENUE & COLLECTIONS
# ===========================================================================
with tab3:
    st.subheader("Revenue & Collections (customer-level, de-duplicated)")

    total_quotation = smart_total(df, "Quotation value")
    total_revenue = smart_total(df, "Total revenue collected")
    total_pending = smart_total(df, "Pending Amount to collect\n (in lacs) without gst")

    kpi_row([
        ("Total Quotation Value", f"₹{total_quotation/1e7:.2f} Cr", None, KPI_COLORS[0]),
        ("Total Revenue Collected", f"₹{total_revenue/1e7:.2f} Cr", None, KPI_COLORS[1]),
        ("Pending to Collect", f"₹{total_pending:.1f} L", None, KPI_COLORS[5]),
    ])

    st.markdown("---")
    st.markdown("**Revenue Collected Trend by Booking Month**")
    cust_month = df.dropna(subset=["Booking Month"]).groupby("Project Parent Code")["Booking Month"] \
        .min().dt.to_period("M").astype(str)
    rev_by_cust = customer_level_value(df, "Total revenue collected")
    trend = pd.DataFrame({"Month": cust_month, "Revenue": rev_by_cust}).dropna() \
        .groupby("Month")["Revenue"].sum().reset_index()
    fig = px.line(trend, x="Month", y="Revenue", markers=True, color_discrete_sequence=[ACCENT])
    fig.update_traces(line=dict(width=3), marker=dict(size=9, color="#E63946"))
    st.plotly_chart(style_fig(fig, title="Revenue Collected, by Booking-Month Cohort"),
                     use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Collection Efficiency by Zone**")
        rows = []
        for z, g in df.groupby("Zone"):
            q, r = smart_total(g, "Quotation value"), smart_total(g, "Total revenue collected")
            rows.append({"Zone": z, "Collection %": safe_pct(r, q)})
        zone_df = pd.DataFrame(rows).sort_values("Collection %", ascending=False)
        fig = px.bar(zone_df, x="Zone", y="Collection %", color_discrete_sequence=[ACCENT], text="Collection %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_color=ACCENT)
        st.plotly_chart(style_fig(fig, showlegend=False, title="Revenue Collected as % of Quotation, by Zone"),
                         use_container_width=True)

    with col_b:
        st.markdown("**Top 10 Localities by Revenue Collected**")
        rows = []
        for loc, g in df.groupby("Locality"):
            rows.append({"Locality": loc, "Revenue Collected": smart_total(g, "Total revenue collected")})
        loc_df = pd.DataFrame(rows).sort_values("Revenue Collected", ascending=False).head(10)
        fig = px.bar(loc_df, x="Revenue Collected", y="Locality", orientation="h",
                     color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(style_fig(fig, showlegend=False, title="Top 10 Localities by Revenue Collected"),
                         use_container_width=True)

    st.markdown("**Pending Amount to Collect — Distribution**")
    st.markdown(
        '<div class="insight-box">📌 <b>What this shows:</b> for every customer, '
        '<code>Pending Amount to collect</code> = value already billed at the current stage minus '
        'what they\'ve actually paid so far (from the source column, GST excluded). '
        'This is an <b>accounts-receivable snapshot</b>, not an aging report — the source data has '
        'no due-date field, so we cannot say how overdue any of it is, only how much is currently '
        'outstanding. Bars to the right of ₹0 are customers who still owe money; a bar at or below '
        '₹0 means they\'ve paid at least what\'s been billed so far.</div>',
        unsafe_allow_html=True,
    )
    pending_series = customer_level_value(
        df, "Pending Amount to collect\n (in lacs) without gst"
    ).reset_index()
    pending_series.columns = ["Project Parent Code", "Pending Amount (Lacs)"]
    fig = px.histogram(pending_series, x="Pending Amount (Lacs)", nbins=30,
                        color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_color=ACCENT)
    st.plotly_chart(style_fig(fig, showlegend=False, title="Customers by Pending Amount Outstanding (Lacs)"),
                     use_container_width=True)

# ===========================================================================
# TAB 4: TAT & AGING
# ===========================================================================
with tab4:
    st.subheader("Turnaround Time & Aging (Operational Efficiency)")
    st.markdown(
        '<div class="insight-box">📌 <b>Definitions:</b><br>'
        '• <b>TAT (months)</b> = average months between two milestone dates '
        '(e.g. Booking → Design Complete), calculated only for bathrooms that have '
        '<u>actually reached both milestones</u>. In-progress bathrooms without an end date '
        'are excluded from the average, not treated as zero.<br>'
        '• <b>Aging bracket (OnTrack / Delayed)</b> is how the company\'s own ops team has '
        'already classified bathrooms still sitting in a stage, using their internal SLA '
        '(the SLA thresholds themselves are not in this dataset — we\'re reporting their '
        'existing classification, not recomputing it).</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Design Aging (from Booking)**")
        d = counts_table(df["Design aging (Bracket) From Booking"], "Bracket", "Count")
        if len(d):
            fig = px.pie(d, names="Bracket", values="Count", hole=0.4, color_discrete_sequence=COLOR_SEQ)
            fig.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(style_fig(fig, title="Design Stage: On Track vs Delayed"), use_container_width=True)
        else:
            st.caption("No data in current filter selection.")
    with col_b:
        st.markdown("**PI Aging (from Booking)**")
        d = counts_table(df["PI aging (Bracket) From Booking"], "Bracket", "Count")
        if len(d):
            fig = px.pie(d, names="Bracket", values="Count", hole=0.4, color_discrete_sequence=COLOR_SEQ)
            fig.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(style_fig(fig, title="PI Stage: On Track vs Delayed"), use_container_width=True)
        else:
            st.caption("No data in current filter selection.")
    with col_c:
        st.markdown("**Execution Aging (from Design payment)**")
        d = counts_table(df["Execution aging (Bracket) From Design payment"], "Bracket", "Count")
        if len(d):
            fig = px.pie(d, names="Bracket", values="Count", hole=0.4, color_discrete_sequence=COLOR_SEQ)
            fig.update_traces(textinfo="label+percent", textfont_size=14)
            st.plotly_chart(style_fig(fig, title="Execution Stage: On Track vs Delayed"), use_container_width=True)
        else:
            st.caption("No data in current filter selection.")

    st.markdown("---")
    avg_design_months = df["Months to Design completed from booking"].mean()
    avg_exec_months = df["Months to Execution completed from Design"].mean()
    n_design_done = int(df["Months to Design completed from booking"].notna().sum())
    n_exec_done = int(df["Months to Execution completed from Design"].notna().sum())

    kpi_row([
        ("Avg Months: Booking → Design Complete",
         f"{avg_design_months:.1f}" if pd.notna(avg_design_months) else "N/A",
         f"based on {n_design_done:,} completed bathrooms", KPI_COLORS[0]),
        ("Avg Months: Design → Execution Complete",
         f"{avg_exec_months:.1f}" if pd.notna(avg_exec_months) else "N/A",
         f"based on {n_exec_done:,} completed bathrooms", KPI_COLORS[1]),
    ])

    st.markdown("**On-Hold Aging Distribution (days)**")
    hold_aging = df[["On hold aging"]].dropna().rename(columns={"On hold aging": "On-Hold Aging (days)"})
    if len(hold_aging):
        fig = px.histogram(hold_aging, x="On-Hold Aging (days)", nbins=30, color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(style_fig(fig, showlegend=False, title="How Long On-Hold Bathrooms Have Been Waiting"),
                         use_container_width=True)
    else:
        st.caption("No on-hold aging data in current filter selection.")

# ===========================================================================
# TAB 5: TEAM PERFORMANCE
# ===========================================================================
with tab5:
    st.subheader("Team Performance")

    rows = []
    for team, g in df.groupby("Sales team"):
        n_bath = len(g)
        n_cust = g["Project Parent Code"].nunique()
        quotation = smart_total(g, "Quotation value")
        revenue = smart_total(g, "Total revenue collected")
        handover_rate = safe_pct((g["Final Project status"] == "Handover Completed").sum(), n_bath)
        lost_rate = safe_pct(g["Final Project status"].isin(
            ["Lost at Design", "Lost at Execution", "Cancelled at sales"]).sum(), n_bath)
        rows.append({
            "Sales Team": team, "Bathrooms Booked": n_bath, "Unique Customers": n_cust,
            "Quotation Value (Cr)": round(quotation/1e7, 2), "Revenue Collected (Cr)": round(revenue/1e7, 2),
            "Handover Rate %": round(handover_rate, 1), "Lost/Cancelled Rate %": round(lost_rate, 1)
        })
    sales_df = pd.DataFrame(rows).sort_values("Bathrooms Booked", ascending=False)

    st.markdown("**Sales Team — Quotation Value Booked**")
    fig = px.bar(sales_df, x="Sales Team", y="Quotation Value (Cr)", text="Bathrooms Booked",
                 color_discrete_sequence=[ACCENT])
    fig.update_traces(marker_color=ACCENT, textposition="outside")
    st.plotly_chart(style_fig(fig, showlegend=False,
                               title="Quotation Value by Sales Rep (label = bathrooms booked)"),
                     use_container_width=True)
    st.dataframe(sales_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Designer — Bathrooms Handled**")
        rows = []
        for d, g in df.groupby("Designer"):
            avg_aging = g["Months to Design completed from booking"].mean()
            rows.append({"Designer": d, "Bathrooms": len(g),
                         "Avg Design Aging (months)": round(avg_aging, 1) if pd.notna(avg_aging) else None})
        designer_df = pd.DataFrame(rows).sort_values("Bathrooms", ascending=False).head(15)
        fig = px.bar(designer_df, x="Designer", y="Bathrooms", color_discrete_sequence=[COLOR_SEQ[2]])
        fig.update_traces(marker_color=COLOR_SEQ[2])
        st.plotly_chart(style_fig(fig, showlegend=False, title="Bathrooms Handled by Designer (Top 15)"),
                         use_container_width=True)
        st.dataframe(designer_df, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**PM — Handover Rate (Execution)**")
        rows = []
        for pm, g in df.groupby("PM"):
            hr = safe_pct((g["Final Project status"] == "Handover Completed").sum(), len(g))
            rows.append({"PM": pm, "Bathrooms": len(g), "Handover Rate %": round(hr, 1)})
        pm_df = pd.DataFrame(rows).sort_values("Bathrooms", ascending=False).head(15)
        fig = px.bar(pm_df, x="PM", y="Handover Rate %", color_discrete_sequence=[COLOR_SEQ[3]])
        fig.update_traces(marker_color=COLOR_SEQ[3])
        st.plotly_chart(style_fig(fig, showlegend=False, title="Handover Rate by PM (Top 15 by volume)"),
                         use_container_width=True)
        st.dataframe(pm_df, hide_index=True, use_container_width=True)

# ===========================================================================
# TAB 6: GEOGRAPHIC — map view (replaces the old grid-table breakdown)
# ===========================================================================
with tab6:
    st.subheader("Geographic Performance")
    st.markdown(
        '<div class="insight-box">📌 <b>Note on locations:</b> the source data has no lat/lon — '
        'bubble positions below use approximate zone-centroid coordinates for Delhi-NCR, with a '
        'small offset per locality so overlapping areas stay readable. Treat this as directional '
        '(which pockets are strong), not a precise address-level map.</div>',
        unsafe_allow_html=True,
    )

    rows = []
    for (zone, loc), g in df.groupby(["Zone", "Locality"]):
        quotation = smart_total(g, "Quotation value")
        revenue = smart_total(g, "Total revenue collected")
        n_cust = g["Project Parent Code"].nunique()
        lat, lon = locality_coords(zone, loc)
        rows.append({
            "Zone": zone, "Locality": loc, "Bathrooms": len(g), "Unique Customers": n_cust,
            "Quotation Value (Cr)": round(quotation/1e7, 2),
            "Revenue Collected (Cr)": round(revenue/1e7, 2),
            "lat": lat, "lon": lon,
        })
    geo_df = pd.DataFrame(rows)

    fig = px.scatter_mapbox(
        geo_df, lat="lat", lon="lon", size="Bathrooms", color="Zone",
        hover_name="Locality",
        hover_data={"Bathrooms": True, "Quotation Value (Cr)": True,
                    "Revenue Collected (Cr)": True, "lat": False, "lon": False},
        color_discrete_sequence=COLOR_SEQ, size_max=38, zoom=8.6,
        center={"lat": 28.58, "lon": 77.22},
    )
    fig.update_layout(mapbox_style="open-street-map", height=560,
                       margin=dict(t=10, b=0, l=0, r=0),
                       legend=dict(font=dict(size=13)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 10 Localities by Bathrooms Booked**")
    top_loc = geo_df.sort_values("Bathrooms", ascending=False).head(10)
    fig = px.bar(top_loc, x="Bathrooms", y="Locality", orientation="h", color="Zone",
                 color_discrete_sequence=COLOR_SEQ)
    st.plotly_chart(style_fig(fig, title="Top 10 Localities by Bathrooms Booked"), use_container_width=True)

    st.markdown("**Zone-wise Summary**")
    zone_summary = geo_df.groupby("Zone").agg(
        Bathrooms=("Bathrooms", "sum"),
        **{"Quotation Value (Cr)": ("Quotation Value (Cr)", "sum")},
        **{"Revenue Collected (Cr)": ("Revenue Collected (Cr)", "sum")},
    ).reset_index().sort_values("Bathrooms", ascending=False)
    st.dataframe(zone_summary, hide_index=True, use_container_width=True)
