"""
MSafe-style Bathroom Renovation - Senior Management Dashboard
Grain: 1 row = 1 bathroom (Project Child Code). Multiple bathrooms can belong
to the same customer (Project Parent Code).

No pivot tables used anywhere - all aggregation via pandas groupby/agg.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Bathroom Renovation - Management Dashboard", layout="wide")

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
    # numeric-with-'-'-placeholder columns -> coerce to numeric
    for c in ["Months to Design completed from booking", "Months to Execution completed from Design"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_raw = load_data()

# ---------------------------------------------------------------------------
# CUSTOMER-LEVEL "SMART SUM" — fixes the repeated-quotation-value data issue
# (51 of 256 multi-bathroom customers have identical Quotation value copy-
# pasted across every bathroom row instead of per-bathroom allocation).
# Rule: if all rows for a customer share one identical value -> count once.
#       if values differ per row -> they are per-bathroom, so sum them.
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
st.title("🚿 Bathroom Renovation Business — Management Dashboard")
st.caption("Grain: 1 row = 1 bathroom. A customer (Project Parent Code) may book multiple bathrooms.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Overview", "🔀 Funnel & Conversion", "💰 Revenue & Collections",
     "⏱️ TAT & Aging", "👥 Team Performance", "🗺️ Geographic"]
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
    collection_pct = (total_revenue / total_quotation * 100) if total_quotation else 0
    total_pending = smart_total(df, "Pending Amount to collect\n (in lacs) without gst")
    handover_pct = (df["Final Project status"] == "Handover Completed").mean() * 100
    lost_cancelled_pct = df["Final Project status"].isin(
        ["Lost at Design", "Lost at Execution", "Cancelled at sales"]).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bathrooms Booked", f"{n_bathrooms:,}")
    c2.metric("Unique Customers", f"{n_customers:,}")
    c3.metric("Avg Bathrooms / Customer", f"{avg_bathrooms_per_cust:.2f}")
    c4.metric("Handover Completed %", f"{handover_pct:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Quotation Value", f"₹{total_quotation/1e7:.2f} Cr")
    c6.metric("Total Revenue Collected", f"₹{total_revenue/1e7:.2f} Cr")
    c7.metric("Collection Efficiency", f"{collection_pct:.1f}%")
    c8.metric("Lost / Cancelled %", f"{lost_cancelled_pct:.1f}%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Bookings Trend by Month**")
        monthly = df.dropna(subset=["Booking Month"]).groupby(
            df["Booking Month"].dt.to_period("M").astype(str)
        ).size().reset_index(name="Bathrooms Booked")
        monthly.columns = ["Month", "Bathrooms Booked"]
        fig = px.bar(monthly, x="Month", y="Bathrooms Booked")
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown("**Final Project Status Distribution**")
        status_counts = df["Final Project status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.bar(status_counts, x="Count", y="Status", orientation="h")
        st.plotly_chart(fig, width='stretch')

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

    booked = len(df)
    reached_design_pay = df["60% Date"].notna().sum()
    reached_final_pay = df["50% date"].notna().sum()
    handed_over = (df["Final Project status"] == "Handover Completed").sum()

    funnel_df = pd.DataFrame({
        "Stage": ["Booked", "Reached Design Payment (60%)", "Reached Final Payment (50%)", "Handover Completed"],
        "Bathrooms": [booked, reached_design_pay, reached_final_pay, handed_over]
    })
    fig = px.funnel(funnel_df, x="Bathrooms", y="Stage")
    st.plotly_chart(fig, width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Booking → Design Payment", f"{reached_design_pay/booked*100:.1f}%" if booked else "0%")
    c2.metric("Design Payment → Final Payment", f"{reached_final_pay/reached_design_pay*100:.1f}%" if reached_design_pay else "0%")
    c3.metric("Final Payment → Handover", f"{handed_over/reached_final_pay*100:.1f}%" if reached_final_pay else "0%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Lost / Cancelled Breakdown**")
        lost = df[df["Final Project status"].isin(
            ["Lost at Design", "Lost at Execution", "Cancelled at sales"])]
        lost_counts = lost["Final Project status"].value_counts().reset_index()
        lost_counts.columns = ["Reason Stage", "Count"]
        fig = px.pie(lost_counts, names="Reason Stage", values="Count", hole=0.4)
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown("**On-Hold Design Reasons**")
        holds = df["On hold Design Reasons"].dropna().value_counts().reset_index()
        holds.columns = ["Reason", "Count"]
        if len(holds):
            fig = px.bar(holds, x="Count", y="Reason", orientation="h")
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No on-hold reasons in current filter selection.")

    st.markdown("**Sales / Design / Execution Stage Breakdown**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Sales Booking Stage")
        st.dataframe(df["Sales Booking Stage"].value_counts().reset_index(name="Count"), hide_index=True)
    with c2:
        st.caption("Design Stage Status")
        st.dataframe(df["Design Stage Status"].value_counts().reset_index(name="Count"), hide_index=True)
    with c3:
        st.caption("Execution Stage Status")
        st.dataframe(df["Execution Stage Status"].value_counts().reset_index(name="Count"), hide_index=True)

# ===========================================================================
# TAB 3: REVENUE & COLLECTIONS
# ===========================================================================
with tab3:
    st.subheader("Revenue & Collections (customer-level, de-duplicated)")

    total_quotation = smart_total(df, "Quotation value")
    total_revenue = smart_total(df, "Total revenue collected")
    total_pending = smart_total(df, "Pending Amount to collect\n (in lacs) without gst")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Quotation Value", f"₹{total_quotation/1e7:.2f} Cr")
    c2.metric("Total Revenue Collected", f"₹{total_revenue/1e7:.2f} Cr")
    c3.metric("Pending to Collect", f"₹{total_pending:.1f} L")

    st.markdown("---")

    st.markdown("**Revenue Collected Trend by Booking Month**")
    monthly_rev = customer_level_value(
        df.assign(_month=df["Booking Month"].dt.to_period("M").astype(str)),
        "Total revenue collected"
    )
    # rebuild with month attached at customer level for trend (first booking month per customer)
    cust_month = df.dropna(subset=["Booking Month"]).groupby("Project Parent Code")["Booking Month"].min().dt.to_period("M").astype(str)
    rev_by_cust = customer_level_value(df, "Total revenue collected")
    trend = pd.DataFrame({"Month": cust_month, "Revenue": rev_by_cust}).dropna().groupby("Month")["Revenue"].sum().reset_index()
    fig = px.line(trend, x="Month", y="Revenue", markers=True)
    st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Collection Efficiency by Zone**")
        rows = []
        for z, g in df.groupby("Zone"):
            q, r = smart_total(g, "Quotation value"), smart_total(g, "Total revenue collected")
            rows.append({"Zone": z, "Collection %": (r/q*100) if q else 0})
        zone_df = pd.DataFrame(rows).sort_values("Collection %", ascending=False)
        fig = px.bar(zone_df, x="Zone", y="Collection %")
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.markdown("**Top 10 Localities by Revenue Collected**")
        rows = []
        for loc, g in df.groupby("Locality"):
            rows.append({"Locality": loc, "Revenue Collected": smart_total(g, "Total revenue collected")})
        loc_df = pd.DataFrame(rows).sort_values("Revenue Collected", ascending=False).head(10)
        fig = px.bar(loc_df, x="Revenue Collected", y="Locality", orientation="h")
        st.plotly_chart(fig, width='stretch')

    st.markdown("**Pending Amount to Collect — Distribution**")
    pending_series = customer_level_value(df, "Pending Amount to collect\n (in lacs) without gst")
    fig = px.histogram(pending_series, nbins=30, labels={"value": "Pending Amount (Lacs)"})
    st.plotly_chart(fig, width='stretch')

# ===========================================================================
# TAB 4: TAT & AGING
# ===========================================================================
with tab4:
    st.subheader("Turnaround Time & Aging (Operational Efficiency)")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Design Aging (from Booking)**")
        d = df["Design aging (Bracket) From Booking"].dropna().value_counts().reset_index()
        d.columns = ["Bracket", "Count"]
        fig = px.pie(d, names="Bracket", values="Count", hole=0.4)
        st.plotly_chart(fig, width='stretch')
    with col_b:
        st.markdown("**PI Aging (from Booking)**")
        d = df["PI aging (Bracket) From Booking"].dropna().value_counts().reset_index()
        d.columns = ["Bracket", "Count"]
        if len(d):
            fig = px.pie(d, names="Bracket", values="Count", hole=0.4)
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No data in current filter selection.")
    with col_c:
        st.markdown("**Execution Aging (from Design payment)**")
        d = df["Execution aging (Bracket) From Design payment"].dropna().value_counts().reset_index()
        d.columns = ["Bracket", "Count"]
        if len(d):
            fig = px.pie(d, names="Bracket", values="Count", hole=0.4)
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No data in current filter selection.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    avg_design_months = df["Months to Design completed from booking"].mean()
    avg_exec_months = df["Months to Execution completed from Design"].mean()
    c1.metric("Avg Months: Booking → Design Complete", f"{avg_design_months:.1f}" if pd.notna(avg_design_months) else "N/A")
    c2.metric("Avg Months: Design → Execution Complete", f"{avg_exec_months:.1f}" if pd.notna(avg_exec_months) else "N/A")

    st.markdown("**On-Hold Aging Distribution (days)**")
    hold_aging = df["On hold aging"].dropna()
    if len(hold_aging):
        fig = px.histogram(hold_aging, nbins=30, labels={"value": "On-Hold Aging (days)"})
        st.plotly_chart(fig, width='stretch')
    else:
        st.caption("No on-hold aging data in current filter selection.")

# ===========================================================================
# TAB 5: TEAM PERFORMANCE
# ===========================================================================
with tab5:
    st.subheader("Team Performance")

    st.markdown("**Sales Team Leaderboard**")
    rows = []
    for team, g in df.groupby("Sales team"):
        n_bath = len(g)
        n_cust = g["Project Parent Code"].nunique()
        quotation = smart_total(g, "Quotation value")
        revenue = smart_total(g, "Total revenue collected")
        handover_rate = (g["Final Project status"] == "Handover Completed").mean() * 100
        lost_rate = g["Final Project status"].isin(
            ["Lost at Design", "Lost at Execution", "Cancelled at sales"]).mean() * 100
        rows.append({
            "Sales Team": team, "Bathrooms Booked": n_bath, "Unique Customers": n_cust,
            "Quotation Value (Cr)": round(quotation/1e7, 2), "Revenue Collected (Cr)": round(revenue/1e7, 2),
            "Handover Rate %": round(handover_rate, 1), "Lost/Cancelled Rate %": round(lost_rate, 1)
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Bathrooms Booked", ascending=False), hide_index=True, width='stretch')

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Designer Leaderboard (by bathrooms handled)**")
        rows = []
        for d, g in df.groupby("Designer"):
            rows.append({
                "Designer": d, "Bathrooms": len(g),
                "Avg Design Aging (months)": round(g["Months to Design completed from booking"].mean(), 1)
                    if pd.notna(g["Months to Design completed from booking"].mean()) else None
            })
        designer_df = pd.DataFrame(rows).sort_values("Bathrooms", ascending=False).head(15)
        st.dataframe(designer_df, hide_index=True, width='stretch')

    with col_b:
        st.markdown("**PM Leaderboard (Execution)**")
        rows = []
        for pm, g in df.groupby("PM"):
            rows.append({"PM": pm, "Bathrooms": len(g),
                         "Handover Rate %": round((g["Final Project status"] == "Handover Completed").mean()*100, 1)})
        pm_df = pd.DataFrame(rows).sort_values("Bathrooms", ascending=False).head(15)
        st.dataframe(pm_df, hide_index=True, width='stretch')

# ===========================================================================
# TAB 6: GEOGRAPHIC
# ===========================================================================
with tab6:
    st.subheader("Geographic Performance")

    st.markdown("**Zone-wise Performance**")
    rows = []
    for z, g in df.groupby("Zone"):
        quotation = smart_total(g, "Quotation value")
        revenue = smart_total(g, "Total revenue collected")
        rows.append({
            "Zone": z, "Bathrooms": len(g), "Unique Customers": g["Project Parent Code"].nunique(),
            "Quotation Value (Cr)": round(quotation/1e7, 2), "Revenue Collected (Cr)": round(revenue/1e7, 2),
            "Avg Ticket Size (Lacs)": round(quotation/1e5/g["Project Parent Code"].nunique(), 1) if g["Project Parent Code"].nunique() else 0
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Bathrooms", ascending=False), hide_index=True, width='stretch')

    st.markdown("**Locality-wise Performance**")
    rows = []
    for loc, g in df.groupby("Locality"):
        quotation = smart_total(g, "Quotation value")
        revenue = smart_total(g, "Total revenue collected")
        rows.append({
            "Locality": loc, "Bathrooms": len(g), "Unique Customers": g["Project Parent Code"].nunique(),
            "Quotation Value (Cr)": round(quotation/1e7, 2), "Revenue Collected (Cr)": round(revenue/1e7, 2)
        })
    loc_df = pd.DataFrame(rows).sort_values("Bathrooms", ascending=False)
    st.dataframe(loc_df, hide_index=True, width='stretch')

    fig = px.bar(loc_df.head(11), x="Locality", y="Bathrooms")
    st.plotly_chart(fig, width='stretch')
