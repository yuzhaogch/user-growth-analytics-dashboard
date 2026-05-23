import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.data_utils import (
    load_orders,
    filter_df,
    compute_growth_kpis,
    compute_retention_metrics,
    compute_funnel_metrics,
    build_user_profile,
    channel_retention,
    cohort_analysis,
)

st.set_page_config(page_title="用户增长与经营分析看板", layout="wide")

DATA_PATH = os.environ.get("ORDERS_CSV", os.path.join(os.path.dirname(__file__), "..", "data", "orders.csv"))


@st.cache_data(show_spinner=False)
def load_data():
    return load_orders(DATA_PATH)


df = load_data()
min_d, max_d = df["order_date"].dt.date.min(), df["order_date"].dt.date.max()

st.title("用户增长与经营分析看板")
st.caption("基于电商订单行为数据构建的增长分析工作台。")

st.sidebar.header("筛选条件")
date_range = st.sidebar.date_input("日期范围", value=(min_d, max_d), min_value=min_d, max_value=max_d)
countries = st.sidebar.multiselect("国家", sorted(df["country"].unique().tolist()))
channels = st.sidebar.multiselect("渠道", sorted(df["channel"].unique().tolist()))
categories = st.sidebar.multiselect("品类", sorted(df["category"].unique().tolist()))

fdf = filter_df(df, date_range, countries, channels, categories)

growth_kpis = compute_growth_kpis(fdf, df)
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("DAU", f"{growth_kpis['DAU']:,}")
col2.metric("新增用户", f"{growth_kpis['New Users']:,}")
col3.metric("下单用户", f"{growth_kpis['Ordering Users']:,}")
col4.metric("支付用户", f"{growth_kpis['Paying Users']:,}")
col5.metric("支付转化率", f"{growth_kpis['Payment Conversion'] * 100:,.1f}%")
col6.metric("次日留存率", f"{growth_kpis['Next-Day Retention'] * 100:,.1f}%")
st.caption("增长指标基于当前筛选范围内的订单行为进行估算。")

st.markdown("---")

ts = fdf.groupby("order_month").agg({"revenue": "sum", "profit": "sum", "order_id": "nunique"}).reset_index()
ts = ts.sort_values("order_month")
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(x=ts["order_month"], y=ts["revenue"], mode="lines+markers", name="营收"))
fig_ts.add_trace(go.Scatter(x=ts["order_month"], y=ts["profit"], mode="lines+markers", name="利润", yaxis="y2"))
fig_ts.update_layout(
    title="月度营收与利润趋势",
    xaxis_title="月份",
    yaxis_title="营收",
    yaxis2=dict(title="利润", overlaying="y", side="right"),
    margin=dict(l=40, r=40, t=60, b=40),
    height=420,
)
st.plotly_chart(fig_ts, use_container_width=True)

c1, c2 = st.columns(2)
cat_rev = fdf.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
fig_cat = px.bar(cat_rev, x="category", y="revenue", title="各品类营收贡献")
c1.plotly_chart(fig_cat, use_container_width=True)

prod = (
    fdf.groupby(["product_id", "subcategory"])["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=False)
    .head(15)
)
fig_prod = px.bar(prod, x="revenue", y="product_id", color="subcategory", title="营收贡献 Top 15 产品", orientation="h")
c2.plotly_chart(fig_prod, use_container_width=True)

geo = fdf.groupby(["country", "city"])["revenue"].sum().reset_index()
fig_geo = px.treemap(geo, path=["country", "city"], values="revenue", title="地理分布营收贡献（国家 -> 城市）")
st.plotly_chart(fig_geo, use_container_width=True)

ch = fdf.groupby("channel")["revenue"].sum().reset_index()
fig_ch = px.pie(ch, values="revenue", names="channel", title="渠道贡献占比", hole=0.45)
st.plotly_chart(fig_ch, use_container_width=True)

st.subheader("转化漏斗")
st.caption("当前漏斗为基于订单行为构建的简化经营漏斗，不包含 visit、register 等原始埋点事件。")
funnel_df = compute_funnel_metrics(fdf, df)
fc1, fc2 = st.columns([1.5, 1.1])

with fc1:
    if funnel_df.empty:
        st.info("当前筛选条件下暂无可计算的漏斗数据。")
    else:
        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_df["stage"],
                x=funnel_df["users"],
                textinfo="value+percent initial",
                marker=dict(color=["#1f4e79", "#2f6fa1", "#4b95c8", "#7bb3d6", "#a7d0e8"]),
            )
        )
        fig_funnel.update_layout(title="用户经营漏斗", margin=dict(l=40, r=40, t=60, b=20), height=420)
        st.plotly_chart(fig_funnel, use_container_width=True)

with fc2:
    if not funnel_df.empty:
        funnel_display = funnel_df.copy()
        funnel_display["整体转化率(%)"] = (funnel_display["conversion_rate"] * 100).round(1)
        funnel_display["环节流失率(%)"] = (funnel_display["dropoff_rate"] * 100).round(1)
        funnel_display = funnel_display[["stage", "users", "整体转化率(%)", "环节流失率(%)"]]
        funnel_display.columns = ["阶段", "用户数", "整体转化率(%)", "环节流失率(%)"]
        st.dataframe(funnel_display, use_container_width=True, hide_index=True)

st.subheader("留存分析")
retention_metrics = compute_retention_metrics(fdf, df)
rc1, rc2, rc3 = st.columns(3)
rc1.metric("新增 cohort 用户", f"{retention_metrics['New Users']:,}")
rc2.metric("次日留存率", f"{retention_metrics['D1 Retention'] * 100:,.1f}%")
rc3.metric("7日留存率", f"{retention_metrics['D7 Retention'] * 100:,.1f}%")
st.caption("留存口径：以首购用户为 cohort，统计首购后第 1 天和第 7 天是否再次活跃。")

channel_ret = channel_retention(fdf, df)
cc1, cc2 = st.columns([1.2, 1.8])

with cc1:
    st.caption("分渠道留存表现")
    if channel_ret.empty:
        st.info("当前筛选条件下暂无可计算的新增 cohort 用户。")
    else:
        channel_display = channel_ret.copy()
        channel_display["d1_retention"] = (channel_display["d1_retention"] * 100).round(1)
        channel_display["d7_retention"] = (channel_display["d7_retention"] * 100).round(1)
        channel_display.columns = ["渠道", "新增用户", "次日留存率(%)", "7日留存率(%)"]
        st.dataframe(channel_display, use_container_width=True, hide_index=True)

with cc2:
    if not channel_ret.empty:
        channel_plot = channel_ret.copy()
        channel_plot["次日留存率"] = channel_plot["d1_retention"] * 100
        channel_plot["7日留存率"] = channel_plot["d7_retention"] * 100
        channel_plot = channel_plot.melt(
            id_vars=["channel", "new_users"],
            value_vars=["次日留存率", "7日留存率"],
            var_name="留存类型",
            value_name="留存率",
        )
        fig_channel_ret = px.bar(
            channel_plot,
            x="channel",
            y="留存率",
            color="留存类型",
            barmode="group",
            title="分渠道次日/7日留存对比",
        )
        fig_channel_ret.update_layout(xaxis_title="渠道", yaxis_title="留存率 (%)")
        st.plotly_chart(fig_channel_ret, use_container_width=True)

cohort_abs, cohort_ret = cohort_analysis(fdf, df)
if not cohort_ret.empty:
    heatmap_values = (cohort_ret * 100).round(1)
    heatmap_values.index = heatmap_values.index.strftime("%Y-%m")
    fig_cohort = px.imshow(
        heatmap_values,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Greens",
        title="Cohort 留存热力图",
        labels={"x": "距首购月份", "y": "首购 cohort", "color": "留存率 (%)"},
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    with st.expander("查看 cohort 明细表"):
        st.dataframe(heatmap_values, use_container_width=True)

st.subheader("用户价值分层")
user_profile = build_user_profile(fdf, df)
if user_profile.empty:
    st.info("当前筛选条件下暂无可计算的用户画像数据。")
else:
    s1, s2, s3 = st.columns(3)
    s1.metric("高价值活跃用户", f"{(user_profile['segment_label'] == '高价值活跃').sum():,}")
    s2.metric("潜力复购用户", f"{(user_profile['segment_label'] == '潜力复购').sum():,}")
    s3.metric("沉默风险用户", f"{(user_profile['segment_label'] == '沉默风险').sum():,}")

    vc1, vc2 = st.columns([1.1, 1.9])

    with vc1:
        segment_counts = user_profile["segment_label"].value_counts().reset_index()
        segment_counts.columns = ["segment_label", "users"]
        fig_segment = px.bar(segment_counts, x="segment_label", y="users", title="各用户分层人数")
        fig_segment.update_layout(xaxis_title="用户分层", yaxis_title="用户数")
        st.plotly_chart(fig_segment, use_container_width=True)

    with vc2:
        scatter_df = user_profile.copy()
        scatter_df["avg_order_value"] = scatter_df["avg_order_value"].round(2)
        fig_profile = px.scatter(
            scatter_df,
            x="order_count",
            y="total_revenue",
            color="segment_label",
            size="avg_order_value",
            hover_data=["customer_id", "acquisition_channel", "primary_country", "activity_tier", "value_tier"],
            title="用户价值分层散点图",
        )
        fig_profile.update_layout(xaxis_title="订单数", yaxis_title="累计营收")
        st.plotly_chart(fig_profile, use_container_width=True)

    st.caption("以下为按用户粒度聚合生成的用户画像表，适合继续扩展用户分层、渠道质量和召回分析。")
    display_profile = user_profile.copy()
    display_profile.columns = [
        "用户ID",
        "首购渠道",
        "主要国家",
        "首购日期",
        "最近下单日期",
        "生命周期",
        "订单数",
        "活跃天数",
        "累计营收",
        "累计利润",
        "客单价",
        "最近活跃距今天数",
        "活跃度分层",
        "价值分层",
        "R分",
        "F分",
        "M分",
        "用户标签",
    ]
    st.dataframe(display_profile, use_container_width=True, hide_index=True)
    st.download_button(
        "下载用户画像表 CSV",
        data=user_profile.to_csv(index=False).encode("utf-8-sig"),
        file_name="user_profiles.csv",
        mime="text/csv",
    )

st.download_button(
    "下载筛选结果 CSV",
    data=fdf.to_csv(index=False).encode("utf-8"),
    file_name="filtered_orders.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("用户增长与经营分析看板 | Streamlit + Plotly | 模拟电商数据集")
