import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -------------------------
# Page config & custom styles
# -------------------------
st.set_page_config(page_title="AI-Driven Early Warning System for MSMEs", layout="wide")

# Custom CSS for KPI cards and table spacing (presentation-ready)
st.markdown(
    """
    <style>
    /* Card container */
    .kpi-container {
        display: flex;
        gap: 12px;
        margin-bottom: 12px;
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
        border-radius: 12px;
        padding: 14px;
        width: 100%;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(0,0,0,0.04);
    }
    .kpi-title { font-size:14px; color:#6b7280; margin-bottom:6px; }
    .kpi-value { font-size:22px; font-weight:700; margin-bottom:4px; }
    .kpi-sub { font-size:12px; color:#6b7280; }
    .kpi-icon { float: right; font-size:22px; opacity:0.9; }

    /* Table spacing */
    .stDataFrame > div {
        padding: 0 !important;
    }

    /* Smaller header */
    .stHeader {
        font-size:20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper to render a polished KPI card inside a Streamlit column
def render_kpi_card(col, title, value, subtitle="", icon="⚡", bg_color="#ffffff"):
    html = f"""
    <div class="kpi-card" style="border-left:6px solid {bg_color};">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div>
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{subtitle}</div>
        </div>
        <div class="kpi-icon">{icon}</div>
      </div>
    </div>
    """
    col.markdown(html, unsafe_allow_html=True)

# -------------------------
# Generate synthetic dataset
# -------------------------
def generate_synthetic_data(days=90, n_meters=50, seed=42):
    np.random.seed(seed)
    end = pd.Timestamp.now().floor('H')
    start = end - pd.Timedelta(days=days)
    rng = pd.date_range(start=start, end=end, freq='H')
    meters = [f"MTR_{i:03d}" for i in range(1, n_meters+1)]

    sectors = ['Textiles', 'Food Processing', 'Auto Components', 'Metal Works', 'Electronics']
    states = ['Tamil Nadu', 'Karnataka', 'Kerala', 'Telangana', 'Andhra Pradesh']
    
    # metadata
    meta = []
    for i, m in enumerate(meters):
        meta.append({
            'meter_id': m,
            'msme_id': f"MSME_{(i%20)+1:03d}",
            'sector': np.random.choice(sectors),
            'location_district': f"District_{(i%12)+1}",
            'location_state': np.random.choice(states),
            'num_employees': int(np.random.choice([5,10,20,30,50,80])),
            'connected_load_kw': round(np.random.uniform(5,150),1)
        })
    meta = pd.DataFrame(meta)

    # consumption data
    rows = []
    for m in meters:
        base = np.random.uniform(0.2, 6.0)
        seasonal = 1 + 0.25*np.sin(np.linspace(0, 3*np.pi, len(rng)))
        daily = 1 + 0.6*np.sin(2 * np.pi * (rng.hour / 24.0))
        vals = np.array(
            np.abs(base * seasonal * daily + np.random.normal(0, 0.15, len(rng))) * (np.random.uniform(0.5,2.5))
        )

        # Inject sick MSME profiles
        rand_val = np.random.rand()
        if rand_val < 0.1:  
            vals = vals * np.random.uniform(0.2, 0.5)  # sustained low consumption
        elif rand_val < 0.15:  
            noise = np.random.normal(0, 1.5, len(rng))  # high fluctuation
            vals = np.abs(vals + noise)

        # occasional spikes/outages
        spikes_idx = np.random.choice(len(rng), size=int(len(rng)*0.004), replace=False)
        vals[spikes_idx] *= np.random.uniform(3,7)
        outage_idx = np.random.choice(len(rng), size=int(len(rng)*0.002), replace=False)
        vals[outage_idx] *= np.random.uniform(0.01,0.1)

        df = pd.DataFrame({'timestamp': rng, 'meter_id': m, 'consumption_kwh': vals})
        rows.append(df)
    consumption = pd.concat(rows, ignore_index=True)

    # events
    events = []
    for _ in range(int(len(rng)*n_meters*0.001)):
        events.append({
            'timestamp': np.random.choice(rng),
            'meter_id': np.random.choice(meters),
            'event_type': np.random.choice(['maintenance','outage','inspection']),
            'severity': np.random.choice(['low','medium','high'])
        })
    events = pd.DataFrame(events)

    return consumption, meta, events

# -------------------------
# Data preparation
# -------------------------
def prepare_data(consumption, metadata):
    consumption['timestamp'] = pd.to_datetime(consumption['timestamp'])
    df = consumption.merge(metadata, on='meter_id', how='left')
    return df

def compute_kpis(df):
    total_energy = df['consumption_kwh'].sum()
    avg_daily = df.set_index('timestamp').resample('D')['consumption_kwh'].sum().mean()
    peak_hour = df.groupby(pd.Grouper(key='timestamp', freq='H'))['consumption_kwh'].sum().idxmax()
    peak_value = df.groupby(pd.Grouper(key='timestamp', freq='H'))['consumption_kwh'].sum().max()
    return {
        'total_energy_kwh': round(total_energy,2),
        'avg_daily_kwh': round(avg_daily,2),
        'peak_hour': str(peak_hour),
        'peak_value_kwh': round(peak_value,2)
    }

# -------------------------
# Modified anomaly detector:
# returns per-meter features + anomaly flag + risk_score (0-100)
# -------------------------
def detect_anomalies(df, contamination=0.05):
    # meter-level summary features
    features = df.groupby('meter_id')['consumption_kwh'].agg(['mean','std','max']).fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)  # -1 anomaly, 1 normal

    # Use decision_function to compute anomaly score (higher => more anomalous after inversion)
    raw_scores = -iso.decision_function(X)  # invert so larger=more anomalous
    # normalize 0..1 then scale to 0..100
    if raw_scores.max() - raw_scores.min() == 0:
        norm = np.zeros_like(raw_scores)
    else:
        norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    risk_score = (norm * 100).round(1)

    features['anomaly'] = preds
    features['anomaly_score'] = raw_scores
    features['risk_score'] = risk_score
    anomalies = features[features['anomaly']==-1].reset_index()  # anomalous meters

    return anomalies, features

# -------------------------
# Small helper: ensure metadata has lat/lon (if not, create synthetic coords for each state)
# -------------------------
def ensure_coordinates(metadata_df):
    md = metadata_df.copy()
    # accept existing columns if present
    if 'lat' in md.columns and 'lon' in md.columns:
        return md

    if 'latitude' in md.columns and 'longitude' in md.columns:
        md = md.rename(columns={'latitude':'lat','longitude':'lon'})
        return md

    # Synthetic bounding boxes for states (approximate lat/lon ranges) - tuned for southern India demo
    state_bounds = {
        'Tamil Nadu':  (8.0, 13.5, 76.0, 80.5),
        'Karnataka':   (11.5, 18.5, 74.0, 78.5),
        'Kerala':      (8.0, 12.0, 74.5, 77.5),
        'Telangana':   (15.0, 19.5, 77.0, 81.0),
        'Andhra Pradesh': (13.0, 19.5, 77.0, 84.0)
    }
    # if state not in dict, use Tamil Nadu bounding box as fallback
    lats = []
    lons = []
    for _, row in md.iterrows():
        st_name = row.get('location_state', '')
        bounds = state_bounds.get(st_name, state_bounds['Tamil Nadu'])
        lat = np.random.uniform(bounds[0], bounds[1])
        lon = np.random.uniform(bounds[2], bounds[3])
        lats.append(lat)
        lons.append(lon)
    md['lat'] = lats
    md['lon'] = lons
    return md

# -------------------------
# Explainability helper: compute recent vs past % change and produce reason text
# -------------------------
def compute_reason_for_flagging(df_view, df_status, anomalies, features, metadata, recent_days=30, past_days=30, risk_threshold=50.0):
    """
    For each msme in df_status, compute:
    - recent_avg: mean consumption in last `recent_days`
    - past_avg: mean consumption in the previous `past_days` (immediately before recent window)
    - pct_change = 100 * (recent_avg - past_avg) / past_avg (if past_avg>0)
    - high_variance: if meter-level std is high (from features)
    - reasons: string describing why flagged
    Returns df_status with extra columns: recent_avg, past_avg, pct_change, flag_reasons
    """
    now = df_view['timestamp'].max()
    recent_start = now - pd.Timedelta(days=recent_days-1)
    past_start = recent_start - pd.Timedelta(days=past_days)
    past_end = recent_start - pd.Timedelta(days=1)

    # compute recent and past mean consumption per msme
    recent = df_view[df_view['timestamp'] >= recent_start].groupby('msme_id')['consumption_kwh'].mean().reset_index().rename(columns={'consumption_kwh':'recent_avg'})
    past = df_view[(df_view['timestamp'] >= past_start) & (df_view['timestamp'] <= past_end)].groupby('msme_id')['consumption_kwh'].mean().reset_index().rename(columns={'consumption_kwh':'past_avg'})

    df_out = df_status.merge(recent, on='msme_id', how='left').merge(past, on='msme_id', how='left')
    df_out['recent_avg'] = df_out['recent_avg'].fillna(0)
    df_out['past_avg'] = df_out['past_avg'].fillna(0)

    def pct_change(r, p):
        if p == 0:
            return np.nan
        return round(100.0 * (r - p) / p, 1)

    df_out['pct_change'] = df_out.apply(lambda row: pct_change(row['recent_avg'], row['past_avg']), axis=1)

    # metric: high variance if any of its meters has std above median*2 (heuristic)
    meter_std = features.reset_index().merge(metadata[['meter_id','msme_id']], on='meter_id', how='left').groupby('msme_id')['std'].max().reset_index().rename(columns={'std':'max_meter_std'})
    median_std = features['std'].median() if 'std' in features.columns else 0.0
    meter_std['high_variance'] = meter_std['max_meter_std'] > (median_std * 2 if median_std>0 else 0)

    df_out = df_out.merge(meter_std[['msme_id','max_meter_std','high_variance']], on='msme_id', how='left')
    df_out['max_meter_std'] = df_out['max_meter_std'].fillna(0)
    df_out['high_variance'] = df_out['high_variance'].fillna(False)

    # anomalous_msmes from anomalies list (meter-level)
    anomalous_meters = anomalies['meter_id'].unique() if not anomalies.empty else []
    anomalous_msmes = metadata[metadata['meter_id'].isin(anomalous_meters)]['msme_id'].unique()

    # produce reason string
    reasons = []
    for _, row in df_out.iterrows():
        reason_parts = []
        if row['msme_id'] in anomalous_msmes:
            reason_parts.append("Anomalous meter profile")
        if (not np.isnan(row['pct_change'])) and (row['pct_change'] <= -30):
            reason_parts.append(f"Sharp drop {row['pct_change']}% vs previous {past_days}d")
        elif (not np.isnan(row['pct_change'])) and (row['pct_change'] >= 50):
            reason_parts.append(f"Sudden spike {row['pct_change']}% vs previous {past_days}d")
        if row['risk_score'] >= risk_threshold:
            reason_parts.append(f"High risk score ({row['risk_score']})")
        if row['high_variance']:
            reason_parts.append("High meter variance")

        if len(reason_parts) == 0:
            reason = "Monitored — no single dominant reason"
        else:
            reason = "; ".join(reason_parts)
        reasons.append(reason)

    df_out['flag_reasons'] = reasons
    # keep only needed columns appended
    return df_out[['msme_id','recent_avg','past_avg','pct_change','max_meter_std','high_variance','flag_reasons']]

# -------------------------
# Small helper: ensure metadata has lat/lon (if not, create synthetic coords for each state)
# (already defined above as ensure_coordinates)
# -------------------------

# -------------------------
# Streamlit App
# -------------------------
st.title("AI-Driven Early Warning System for MSMEs")
st.markdown("🚀 **Purpose:** Detect vulnerable MSMEs early by leveraging electricity consumption & Udyam metadata, classify them as Healthy/At Risk, and empower government & supervisors with actionable insights for proactive intervention.")

# Upload section
with st.expander("Upload proxy CSVs (optional)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        upload_consumption = st.file_uploader("Upload consumption.csv", type=['csv'], key='c1')
    with col2:
        upload_meta = st.file_uploader("Upload metadata.csv", type=['csv'], key='c2')
    with col3:
        upload_events = st.file_uploader("Upload events.csv (optional)", type=['csv'], key='c3')

# -------------------------
# DATE RANGE (value-add): allow user to select analysis window
# -------------------------
# We'll populate default range after data is loaded below.
# For now set placeholders
date_min_placeholder = pd.Timestamp.now() - pd.Timedelta(days=120)
date_max_placeholder = pd.Timestamp.now()

if upload_consumption and upload_meta:
    consumption = pd.read_csv(upload_consumption)
    metadata = pd.read_csv(upload_meta)
    events = pd.read_csv(upload_events) if upload_events else pd.DataFrame()
else:
    consumption, metadata, events = generate_synthetic_data(days=120, n_meters=160)

# Keep original data merge logic
df = prepare_data(consumption, metadata)

# Now create date range filter UI using actual data range
data_min = df['timestamp'].min()
data_max = df['timestamp'].max()
if pd.isna(data_min) or pd.isna(data_max):
    data_min = date_min_placeholder
    data_max = date_max_placeholder

st.sidebar.header('View / Filters')
# Date range picker (value-add)
start_date, end_date = st.sidebar.date_input("Date range", value=(data_min.date(), data_max.date()))
# convert to timestamps (include full end day)
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)

role = st.sidebar.selectbox('Select Dashboard', ['MSME Owner', 'Government', 'Supervisor'])
state_filter = st.sidebar.multiselect('State', options=sorted(df['location_state'].unique()), default=['Tamil Nadu'])
sector_filter = st.sidebar.multiselect('Sector', options=sorted(df['sector'].unique()), default=sorted(df['sector'].unique()))

# Filter df by date, state, sector (non-invasive)
df_view = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
df_view = df_view[df_view['location_state'].isin(state_filter) & df_view['sector'].isin(sector_filter)]

# -------------------------
# Helper chart
# -------------------------
def timeseries_consumption(df, title='Consumption (kWh)'):
    ts = df.groupby(pd.Grouper(key='timestamp', freq='D'))['consumption_kwh'].sum().reset_index()
    fig = px.line(ts, x='timestamp', y='consumption_kwh', title=title)
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    return fig

# Color palette for charts (consistent)
COLOR_HEALTH = {"Healthy": "#16a34a", "At Risk": "#ef4444"}  # green/red

# -------------------------
# Role-based dashboards (kept your original structure; only Government section enhanced visually + map + explanations)
# -------------------------
if role == 'MSME Owner':
    st.header('MSME Owner — Operational & Cost Insights')
    selected_msme = st.selectbox('Select MSME', options=df_view['msme_id'].unique())
    msme_df = df_view[df_view['msme_id']==selected_msme]
    kpi = compute_kpis(msme_df)
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    c1.metric('Total Energy (kWh)', kpi['total_energy_kwh'])
    c2.metric('Avg Daily (kWh)', kpi['avg_daily_kwh'])
    c3.metric('Peak Hour', kpi['peak_hour'])
    c4.metric('Peak Value (kWh)', kpi['peak_value_kwh'])

    st.plotly_chart(timeseries_consumption(msme_df, title=f'{selected_msme} — Daily Energy Usage'), use_container_width=True)

    hourly = msme_df.set_index('timestamp').groupby(pd.Grouper(freq='H'))['consumption_kwh'].sum().reset_index()
    hourly['hour'] = hourly['timestamp'].dt.hour
    load_profile = hourly.groupby('hour')['consumption_kwh'].mean().reset_index()
    fig_lp = px.bar(load_profile, x='hour', y='consumption_kwh', title='Average Hourly Load Profile')
    st.plotly_chart(fig_lp, use_container_width=True)

    anomalies, _ = detect_anomalies(msme_df)
    st.subheader('Anomalies / Irregularities')
    st.dataframe(anomalies.head(10))

    st.subheader('Top Actionable Recommendations')
    st.markdown("""
    - Shift discretionary loads away from peak hours.
    - Perform preventive maintenance on meters flagged for abrupt drops.
    - Consider demand-side management & energy-efficient motors.
    """)

    # -------------------------
    # Value-add: Simple 7-day predictive trendline (linear extrapolation on daily totals)
    # -------------------------
    if not msme_df.empty:
        st.subheader("7-day Simple Forecast (linear fit on daily totals)")
        daily = msme_df.groupby(pd.Grouper(key='timestamp', freq='D'))['consumption_kwh'].sum().reset_index().dropna()
        if len(daily) >= 3:
            # fit linear trend on days since start
            daily = daily.reset_index(drop=True)
            daily['day_idx'] = (daily['timestamp'] - daily['timestamp'].min()).dt.days
            x = daily['day_idx'].values
            y = daily['consumption_kwh'].values
            # linear fit
            coefs = np.polyfit(x, y, 1)
            poly = np.poly1d(coefs)
            last_day = x.max()
            future_idx = np.arange(last_day+1, last_day+8)
            preds = poly(future_idx)
            # create plot
            future_dates = [daily['timestamp'].max() + pd.Timedelta(days=int(i)) for i in range(1,8)]
            df_future = pd.DataFrame({'timestamp': future_dates, 'consumption_kwh': preds})
            df_plot = pd.concat([daily[['timestamp','consumption_kwh']], df_future], ignore_index=True)
            fig_fore = px.line(df_plot, x='timestamp', y='consumption_kwh', title='Observed + 7-day linear forecast')
            fig_fore.add_scatter(x=df_future['timestamp'], y=df_future['consumption_kwh'], mode='markers', name='Forecast')
            st.plotly_chart(fig_fore, use_container_width=True)
        else:
            st.info("Not enough daily data to produce a robust forecast (need >= 3 days).")

elif role == 'Government':
    st.header('Government — Policy, Monitoring & Compliance')

    # Aggregate by state
    agg_state = df_view.groupby('location_state')['consumption_kwh'].sum().reset_index().sort_values('consumption_kwh', ascending=False)
    fig_state = px.bar(agg_state, x='location_state', y='consumption_kwh', title='Top States by Consumption',
                       color_discrete_sequence=["#0ea5e9"])
    st.plotly_chart(fig_state, use_container_width=True)

    # State > District treemap
    heat = df_view.groupby(['location_state','location_district'])['consumption_kwh'].sum().reset_index()
    heat = heat.sort_values('consumption_kwh', ascending=False).head(50)
    fig_heat = px.treemap(heat, path=['location_state','location_district'], values='consumption_kwh', title='Treemap: State > District Consumption')
    st.plotly_chart(fig_heat, use_container_width=True)

    # Healthy vs At-Risk classification (enhanced)
    anomalies, features = detect_anomalies(df_view, contamination=0.05)

    # map meter-level risk -> msme-level (take max risk of its meters as conservative)
    # ensure metadata includes any extra Udyam fields (do not alter metadata if they exist)
    # we intentionally accept and pass through any additional columns present in uploaded metadata
    meter_features = features.reset_index().merge(metadata[['meter_id','msme_id']], on='meter_id', how='left')
    msme_risk = meter_features.groupby('msme_id')['risk_score'].max().reset_index()

    # base status dataframe (mean consumption per MSME) + risk_score
    df_status = df_view.groupby('msme_id').agg({'consumption_kwh':'mean'}).reset_index()
    df_status = df_status.merge(msme_risk, on='msme_id', how='left').fillna({'risk_score':0})

    # any MSME that has an anomalous meter → flagged, or risk_score threshold
    anomalous_meters = anomalies['meter_id'].unique()
    anomalous_msmes = metadata[metadata['meter_id'].isin(anomalous_meters)]['msme_id'].unique()
    RISK_THRESHOLD = 50.0  # risk_score >= 50 considered At Risk (tuneable)

    df_status['status'] = df_status.apply(
        lambda row: 'At Risk' if (row['msme_id'] in anomalous_msmes) or (row['risk_score'] >= RISK_THRESHOLD) else 'Healthy',
        axis=1
    )

    # -------------------------
    # Value-add: compute reason-for-flagging and time-window metrics
    # -------------------------
    reason_df = compute_reason_for_flagging(df_view, df_status, anomalies, features, metadata, recent_days=30, past_days=30, risk_threshold=RISK_THRESHOLD)
    # merge reason into df_status
    df_status = df_status.merge(reason_df, on='msme_id', how='left').fillna({'recent_avg':0,'past_avg':0,'pct_change':np.nan,'max_meter_std':0,'high_variance':False,'flag_reasons':"Monitored - no dominant reason"})

    # --- Polished KPI CARDS (presentation-ready) ---
    total_msmes = len(df_status)
    healthy_count = len(df_status[df_status['status'] == "Healthy"])
    at_risk_count = len(df_status[df_status['status'] == "At Risk"])
    anomaly_pct = (at_risk_count / total_msmes) * 100 if total_msmes > 0 else 0

    k1,k2,k3,k4 = st.columns([1,1,1,1])
    # polished cards (icons + colored left border)
    render_kpi_card(k1, "🏭 Total MSMEs", f"{total_msmes}", subtitle="Active MSMEs in view", icon="🏢", bg_color="#0ea5e9")
    render_kpi_card(k2, "🟢 Healthy MSMEs", f"{healthy_count}", subtitle="No immediate action required", icon="✅", bg_color=COLOR_HEALTH["Healthy"])
    render_kpi_card(k3, "🔴 At-Risk MSMEs", f"{at_risk_count}", subtitle="Recommend field inspection", icon="⚠️", bg_color=COLOR_HEALTH["At Risk"])
    render_kpi_card(k4, "⚠️ Anomaly %", f"{anomaly_pct:.2f}%", subtitle="Share of MSMEs flagged", icon="📈", bg_color="#f59e0b")

    # --- Small spacer ---
    st.markdown("<br>", unsafe_allow_html=True)

    # Pie chart (color-coded)
    status_counts = df_status['status'].value_counts().reset_index()
    status_counts.columns = ['status','count']
    fig_status = px.pie(status_counts, names='status', values='count', title='MSME Health Status',
                        color='status', color_discrete_map=COLOR_HEALTH)
    fig_status.update_traces(textinfo='percent+label', hole=0.35)
    st.plotly_chart(fig_status, use_container_width=True)

    # Styled Table with row-wise highlighting and risk score column and reason column
    display_cols = ['msme_id','consumption_kwh','risk_score','status','pct_change','flag_reasons']
    def highlight_status(row):
        color = '#fff1f0' if row['status'] == 'At Risk' else '#f0fff4'
        return [f'background-color: {color}'] * len(row)

    styled_df = df_status[display_cols].copy()
    # format pct_change to show +/-
    styled = styled_df.style.apply(highlight_status, axis=1).format({'consumption_kwh':'{:.2f}','risk_score':'{:.1f}','pct_change':'{:+.1f}'})
    st.subheader('MSME Health Table (with Flags, Risk Score & Reason)')
    st.dataframe(styled, use_container_width=True)

    # ✅ Report Download Button (CSV) - includes reason and pct_change
    csv = df_status[display_cols].to_csv(index=False)
    st.download_button("📥 Download MSME Health Report (CSV)", data=csv, file_name="msme_health_report.csv", mime="text/csv")

    st.subheader('Hotspot MSMEs (Anomalous profiles)')
    st.dataframe(anomalies.merge(metadata, on='meter_id', how='left').sort_values('risk_score', ascending=False).head(20), use_container_width=True)

    # -------------------------
    # NEW FEATURE: Interactive Map View (msme risk map)
    # -------------------------
    st.subheader("Interactive Map View — MSME Risk Zones")
    # prepare metadata with coords (either existing lat/lon or synthetic)
    metadata_with_coords = ensure_coordinates(metadata)

    # merge msme-level info (one row per msme) with a representative lat/lon
    meter_md = metadata_with_coords.copy()
    msme_coords = meter_md.groupby('msme_id').agg({'lat':'median','lon':'median'}).reset_index()

    # combine df_status with msme_coords and other contextual fields (district/state)
    msme_map = df_status.merge(msme_coords, on='msme_id', how='left')
    msme_context = metadata_with_coords.groupby('msme_id').agg({'location_state':'first','location_district':'first'}).reset_index()
    msme_map = msme_map.merge(msme_context, on='msme_id', how='left')

    # If any missing lat/lon, generate randomly (fallback)
    missing_coords = msme_map['lat'].isna() | msme_map['lon'].isna()
    if missing_coords.any():
        tmp_coords = ensure_coordinates(msme_map[['msme_id','location_state','location_district']].assign(meter_id=msme_map['msme_id']))
        msme_map.loc[missing_coords, 'lat'] = tmp_coords.loc[missing_coords, 'lat'].values
        msme_map.loc[missing_coords, 'lon'] = tmp_coords.loc[missing_coords, 'lon'].values

    # prepare pydeck layer: circle layer sized by risk_score, colored by status
    size_scale = 200  # visual scale; adjust if points too small/large
    msme_map['point_size'] = (msme_map['risk_score'].fillna(0) / 100.0) * size_scale + 20

    # color map
    def hex_to_rgb(hex_color):
        h = hex_color.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    msme_map['color'] = msme_map['status'].map(COLOR_HEALTH).fillna('#999999')
    msme_map['r'], msme_map['g'], msme_map['b'] = zip(*msme_map['color'].apply(hex_to_rgb))

    # drop rows without coordinates
    msme_map = msme_map.dropna(subset=['lat','lon'])

    if msme_map.empty:
        st.info("No MSME coordinates available to show on map.")
    else:
        # set initial view to center on mean lat/lon
        mid_lat = float(msme_map['lat'].mean())
        mid_lon = float(msme_map['lon'].mean())
        initial_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6, pitch=30)

        tooltip = {
            "html": "<b>MSME ID:</b> {msme_id} <br/> <b>Status:</b> {status} <br/> <b>Risk:</b> {risk_score} <br/> <b>District:</b> {location_district} <br/> <b>State:</b> {location_state} <br/> <b>Reason:</b> {flag_reasons}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=msme_map,
            pickable=True,
            get_position='[lon, lat]',
            get_fill_color=['r','g','b'],
            get_radius='point_size',
            radius_scale=1,
            radius_min_pixels=4,
            radius_max_pixels=200,
            auto_highlight=True
        )

        deck = pdk.Deck(layers=[layer], initial_view_state=initial_view, tooltip=tooltip)
        st.pydeck_chart(deck, use_container_width=True)

    # -------------------------

    st.subheader('Policy Recommendations')
    st.markdown("""
    - 🎯 Target subsidy programs in districts with high anomalies.
    - 🧾 Focus audits where connected load is low but consumption is high.
    - 🔎 Prioritize field inspections for flagged MSMEs before they become NPA.
    """)

elif role == 'Supervisor':
    st.header('Supervisor — Field Operations & Maintenance')
    sel_state = st.selectbox('State', options=sorted(df_view['location_state'].unique()))
    sel_district = st.selectbox('District', options=sorted(df_view[df_view['location_state']==sel_state]['location_district'].unique()))
    sup_df = df_view[(df_view['location_state']==sel_state) & (df_view['location_district']==sel_district)]

    anomalies, _ = detect_anomalies(sup_df, contamination=0.05)
    st.subheader('Recent anomaly list (meters)')
    st.dataframe(anomalies.head(50))

    meter_list = sup_df['meter_id'].unique().tolist()
    chosen_meter = st.selectbox('Choose meter', options=meter_list)
    meter_df = sup_df[sup_df['meter_id']==chosen_meter]
    last_30 = meter_df[meter_df['timestamp'] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]
    if not last_30.empty:
        fig_m = px.line(last_30, x='timestamp', y='consumption_kwh', title=f'{chosen_meter} — Last 30 days')
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader('Inspection Scheduler')
    inspector = st.selectbox('Assign to', options=['Supervisor A','Supervisor B','Field Engineer 1','Field Engineer 2'])
    sched_date = st.date_input('Inspection date', value=pd.Timestamp.now().date())
    notes = st.text_area('Notes')
    if st.button('Schedule Inspection'):
        st.success(f'Inspection for {chosen_meter} scheduled on {sched_date} assigned to {inspector}.')

    st.download_button('📥 Download Meter Summary (CSV)', data=sup_df.to_csv(index=False), file_name='meter_summary.csv', mime='text/csv')
