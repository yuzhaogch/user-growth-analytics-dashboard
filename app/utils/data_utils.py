import pandas as pd
import numpy as np


def load_orders(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["order_date"])
    df["order_month"] = df["order_date"].values.astype("datetime64[M]")
    df["profit"] = df["revenue"] - df["cost"]
    return df


def filter_df(df, date_range, countries=None, channels=None, categories=None):
    mask = (df["order_date"].dt.date >= date_range[0]) & (df["order_date"].dt.date <= date_range[1])
    if countries:
        mask &= df["country"].isin(countries)
    if channels:
        mask &= df["channel"].isin(channels)
    if categories:
        mask &= df["category"].isin(categories)
    return df.loc[mask].copy()


def compute_kpis(df):
    kpis = {}
    kpis["Revenue"] = df["revenue"].sum()
    kpis["Profit"] = df["profit"].sum()
    kpis["Orders"] = df["order_id"].nunique()
    kpis["Customers"] = df["customer_id"].nunique()
    kpis["AOV"] = df.groupby("order_id")["revenue"].sum().mean()
    kpis["Margin%"] = (df["profit"].sum() / df["revenue"].sum()) if df["revenue"].sum() > 0 else 0
    return kpis


def _prepare_retention_scope(df, base_df=None):
    if df.empty:
        return None

    scoped = df.copy()
    reference = base_df.copy() if base_df is not None else scoped.copy()

    scoped["order_day"] = scoped["order_date"].dt.normalize()
    reference["order_day"] = reference["order_date"].dt.normalize()
    reference["order_month"] = reference["order_date"].values.astype("datetime64[M]")

    first_touch = (
        reference.sort_values("order_date")
        .groupby("customer_id")
        .agg(first_order_day=("order_day", "first"), acquisition_channel=("channel", "first"))
    )

    customer_days = reference.groupby("customer_id")["order_day"].agg(set)
    scoped_customers = set(scoped["customer_id"].unique())
    scope_start = scoped["order_day"].min()
    scope_end = scoped["order_day"].max()

    new_users = first_touch[
        first_touch.index.isin(scoped_customers)
        & (first_touch["first_order_day"] >= scope_start)
        & (first_touch["first_order_day"] <= scope_end)
    ].copy()

    if new_users.empty:
        new_users["d1_retained"] = pd.Series(dtype=bool)
        new_users["d7_retained"] = pd.Series(dtype=bool)
        return {
            "scoped": scoped,
            "reference": reference,
            "new_users": new_users,
            "customer_days": customer_days,
        }

    new_users["d1_retained"] = [
        first_day + pd.Timedelta(days=1) in customer_days[customer_id]
        for customer_id, first_day in new_users["first_order_day"].items()
    ]
    new_users["d7_retained"] = [
        first_day + pd.Timedelta(days=7) in customer_days[customer_id]
        for customer_id, first_day in new_users["first_order_day"].items()
    ]

    return {
        "scoped": scoped,
        "reference": reference,
        "new_users": new_users,
        "customer_days": customer_days,
    }


def compute_growth_kpis(df, base_df=None):
    kpis = {
        "DAU": 0,
        "New Users": 0,
        "Ordering Users": 0,
        "Paying Users": 0,
        "Payment Conversion": 0.0,
        "Next-Day Retention": 0.0,
    }

    if df.empty:
        return kpis

    scope = _prepare_retention_scope(df, base_df)
    scoped = scope["scoped"]
    new_users = scope["new_users"]

    latest_day = scoped["order_day"].max()
    kpis["DAU"] = scoped.loc[scoped["order_day"] == latest_day, "customer_id"].nunique()
    kpis["New Users"] = new_users.shape[0]

    ordering_users = scoped["customer_id"].nunique()
    paying_users = scoped.loc[scoped["revenue"] > 0, "customer_id"].nunique()
    kpis["Ordering Users"] = ordering_users
    kpis["Paying Users"] = paying_users
    kpis["Payment Conversion"] = (paying_users / ordering_users) if ordering_users > 0 else 0.0

    if not new_users.empty:
        kpis["Next-Day Retention"] = new_users["d1_retained"].mean()

    return kpis


def compute_retention_metrics(df, base_df=None):
    metrics = {
        "New Users": 0,
        "D1 Retained Users": 0,
        "D7 Retained Users": 0,
        "D1 Retention": 0.0,
        "D7 Retention": 0.0,
    }

    scope = _prepare_retention_scope(df, base_df)
    if scope is None:
        return metrics

    new_users = scope["new_users"]
    if new_users.empty:
        return metrics

    metrics["New Users"] = int(new_users.shape[0])
    metrics["D1 Retained Users"] = int(new_users["d1_retained"].sum())
    metrics["D7 Retained Users"] = int(new_users["d7_retained"].sum())
    metrics["D1 Retention"] = float(new_users["d1_retained"].mean())
    metrics["D7 Retention"] = float(new_users["d7_retained"].mean())
    return metrics


def channel_retention(df, base_df=None):
    scope = _prepare_retention_scope(df, base_df)
    columns = ["channel", "new_users", "d1_retention", "d7_retention"]
    if scope is None or scope["new_users"].empty:
        return pd.DataFrame(columns=columns)

    channel_df = (
        scope["new_users"]
        .groupby("acquisition_channel")
        .agg(
            new_users=("acquisition_channel", "size"),
            d1_retention=("d1_retained", "mean"),
            d7_retention=("d7_retained", "mean"),
        )
        .reset_index()
        .rename(columns={"acquisition_channel": "channel"})
        .sort_values("new_users", ascending=False)
    )
    return channel_df


def compute_funnel_metrics(df, base_df=None):
    columns = ["stage", "users", "conversion_rate", "dropoff_rate"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    scope = _prepare_retention_scope(df, base_df)
    scoped = scope["scoped"]
    reference = scope["reference"]
    new_users = scope["new_users"]

    ordering_users = set(scoped["customer_id"].unique())
    paying_users = set(scoped.loc[scoped["revenue"] > 0, "customer_id"].unique())

    order_counts = scoped.groupby("customer_id")["order_id"].nunique()
    repeat_users = set(order_counts[order_counts >= 2].index)

    customer_revenue = scoped.groupby("customer_id")["revenue"].sum()
    high_value_threshold = customer_revenue.quantile(0.75) if not customer_revenue.empty else 0
    high_value_users = set(customer_revenue[customer_revenue >= high_value_threshold].index) if high_value_threshold > 0 else set()

    stages = [
        ("新增用户", set(new_users.index)),
        ("下单用户", ordering_users),
        ("支付用户", paying_users),
        ("复购用户", repeat_users),
        ("高价值用户", high_value_users),
    ]

    top_count = len(stages[0][1])
    rows = []
    previous_count = None
    for stage_name, users in stages:
        current_count = len(users)
        conversion_rate = (current_count / top_count) if top_count > 0 else 0.0
        dropoff_rate = 0.0
        if previous_count and previous_count > 0:
            dropoff_rate = 1 - (current_count / previous_count)
        rows.append(
            {
                "stage": stage_name,
                "users": current_count,
                "conversion_rate": conversion_rate,
                "dropoff_rate": dropoff_rate,
            }
        )
        previous_count = current_count

    return pd.DataFrame(rows)


def _safe_qcut(series, labels):
    if series.empty:
        return pd.Series(dtype=object, index=series.index)
    pct_rank = series.rank(method="average", pct=True)
    bins = np.linspace(0, 1, len(labels) + 1)
    return pd.cut(pct_rank, bins=bins, labels=labels, include_lowest=True)


def build_user_profile(df, base_df=None):
    columns = [
        "customer_id",
        "acquisition_channel",
        "primary_country",
        "first_order_day",
        "last_order_day",
        "lifecycle_stage",
        "order_count",
        "active_days",
        "total_revenue",
        "total_profit",
        "avg_order_value",
        "recency_days",
        "activity_tier",
        "value_tier",
        "r_score",
        "f_score",
        "m_score",
        "segment_label",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    scope = _prepare_retention_scope(df, base_df)
    scoped = scope["scoped"]
    reference = scope["reference"]
    scope_end = scoped["order_day"].max()

    first_touch = (
        reference.sort_values("order_date")
        .groupby("customer_id")
        .agg(
            first_order_day=("order_day", "first"),
            acquisition_channel=("channel", "first"),
        )
    )

    user_profile = (
        scoped.groupby("customer_id")
        .agg(
            order_count=("order_id", "nunique"),
            active_days=("order_day", "nunique"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            last_order_day=("order_day", "max"),
            primary_country=("country", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        )
        .merge(first_touch, left_index=True, right_index=True, how="left")
        .reset_index()
    )

    user_profile["avg_order_value"] = user_profile["total_revenue"] / user_profile["order_count"]
    user_profile["recency_days"] = (scope_end - user_profile["last_order_day"]).dt.days
    user_profile["days_since_first_order"] = (scope_end - user_profile["first_order_day"]).dt.days

    user_profile["lifecycle_stage"] = np.where(
        user_profile["days_since_first_order"] <= 30,
        "新用户",
        "老用户",
    )

    user_profile["activity_tier"] = pd.cut(
        user_profile["order_count"],
        bins=[0, 1, 3, np.inf],
        labels=["低活跃", "中活跃", "高活跃"],
        include_lowest=True,
    ).astype(str)

    user_profile["value_tier"] = _safe_qcut(
        user_profile["total_revenue"],
        labels=["低价值", "中价值", "中高价值", "高价值"],
    ).astype(str)

    user_profile["r_score"] = _safe_qcut(user_profile["recency_days"], labels=[3, 2, 1]).astype(int)
    user_profile["f_score"] = _safe_qcut(user_profile["order_count"], labels=[1, 2, 3]).astype(int)
    user_profile["m_score"] = _safe_qcut(user_profile["total_revenue"], labels=[1, 2, 3]).astype(int)

    user_profile["segment_label"] = "常规经营"
    user_profile.loc[
        (user_profile["value_tier"] == "高价值") & user_profile["activity_tier"].isin(["中活跃", "高活跃"]),
        "segment_label",
    ] = "高价值活跃"
    user_profile.loc[
        (user_profile["lifecycle_stage"] == "新用户") & (user_profile["order_count"] == 1),
        "segment_label",
    ] = "新客待转化"
    user_profile.loc[
        (user_profile["recency_days"] >= 60) & (user_profile["order_count"] <= 2),
        "segment_label",
    ] = "沉默风险"
    user_profile.loc[
        (user_profile["order_count"] >= 2) & user_profile["segment_label"].eq("常规经营"),
        "segment_label",
    ] = "潜力复购"

    return user_profile[columns].sort_values(
        ["segment_label", "total_revenue"],
        ascending=[True, False],
    )


def cohort_analysis(df, base_df=None):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    reference = base_df.copy() if base_df is not None else df.copy()
    reference["order_month"] = reference["order_date"].values.astype("datetime64[M]")

    first = reference.groupby("customer_id")["order_month"].min().rename("cohort_month")
    tmp = df.merge(first, on="customer_id", how="left")
    tmp["cohort_index"] = (
        (tmp["order_month"].dt.year - tmp["cohort_month"].dt.year) * 12
        + (tmp["order_month"].dt.month - tmp["cohort_month"].dt.month)
        + 1
    )
    cohort = tmp.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()
    cohort_pivot = (
        cohort.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
        .fillna(0)
        .astype(int)
    )
    cohort_ret = cohort_pivot.divide(cohort_pivot[1], axis=0).round(3)
    return cohort_pivot, cohort_ret


def rfm_segmentation(df, as_of=None):
    if as_of is None:
        as_of = df["order_date"].max().normalize() + pd.Timedelta(days=1)
    recency = df.groupby("customer_id")["order_date"].max().apply(lambda d: (as_of - d).days)
    frequency = df.groupby("customer_id")["order_id"].nunique()
    monetary = df.groupby("customer_id")["revenue"].sum()
    r = pd.qcut(recency, 3, labels=[3, 2, 1])
    f = pd.qcut(frequency.rank(method="first"), 3, labels=[1, 2, 3])
    m = pd.qcut(monetary.rank(method="first"), 3, labels=[1, 2, 3])
    rfm = pd.DataFrame({"R": r.astype(int), "F": f.astype(int), "M": m.astype(int)})
    rfm["RFM_Score"] = rfm.sum(axis=1)
    rfm["Segment"] = pd.cut(
        rfm["RFM_Score"],
        bins=[2, 5, 7, 9],
        labels=["New/Cold", "Active", "Champions"],
        include_lowest=True,
    )
    rfm.index.name = "customer_id"
    return rfm.reset_index()
