import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="📈 Forecast eCommerce 2025 – Pełna analiza", layout="wide")
st.title("🛒 Prognoza zamówień eCommerce 2025 z analizą wzrostów")

# === Wczytanie danych ===
uploaded_file = st.sidebar.file_uploader("Wgraj dane (CSV/XLSX)", type=['csv', 'xlsx'])
if uploaded_file is None:
    st.info("📁 Wgraj dane z kolumnami: data, liczba zamówień")
    st.stop()

if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# === Wybór kolumn ===
st.sidebar.header("📌 Kolumny")
date_col = st.sidebar.selectbox("Kolumna z datą", df.columns)
val_col = st.sidebar.selectbox("Kolumna z wartością", [c for c in df.columns if c != date_col])

# === Agregacja ===
agg_type = st.sidebar.selectbox("Agregacja", ["Dzienna", "Tygodniowa", "Miesięczna"], index=2)
ma_window = st.sidebar.slider("Średnia krocząca", 3, 30, 7)

# === Przygotowanie danych ===
data = df[[date_col, val_col]].copy()
data.columns = ['date', 'orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

# === Resampling ===
if agg_type == "Dzienna":
    ts = data.set_index('date')['orders'].resample('D').sum()
    freq_rule, seasonal_periods = 'D', 7
elif agg_type == "Tygodniowa":
    ts = data.set_index('date')['orders'].resample('W-MON').sum()
    freq_rule, seasonal_periods = 'W', 52
else:
    ts = data.set_index('date')['orders'].resample('M').sum()
    freq_rule, seasonal_periods = 'M', 12

st.subheader("📅 Dane wejściowe")
st.write(f"Zakres danych: **{ts.index.min().date()} – {ts.index.max().date()}** ({len(ts)} punktów)")
st.dataframe(ts.tail().rename('orders'))

# === Modelowanie ===
st.subheader("🤖 Model prognozujący")

# Bierzemy tylko do końca 2024
train = ts[ts.index < '2025-01-01']

if len(train) < 3:
    st.error("Za mało danych do modelowania (potrzeba przynajmniej kilku miesięcy historii).")
    st.stop()

# Exponential Smoothing z sezonowością
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
fit = model.fit(optimized=True)

# === Prognoza tylko na 2025 ===
forecast_index = pd.date_range('2025-01-01', '2025-12-31', freq=freq_rule)
forecast = fit.forecast(len(forecast_index))
forecast.index = forecast_index

# === Metryki wzrostów ===
df_forecast = forecast.to_frame('forecast')
df_forecast['dod'] = df_forecast['forecast'].pct_change() * 100
df_forecast['wow'] = df_forecast['forecast'].pct_change(7) * 100 if agg_type == 'Dzienna' else np.nan
df_forecast['mom'] = df_forecast['forecast'].pct_change(1) * 100 if agg_type == 'Miesięczna' else np.nan

# Średnie wzrosty
mean_dod = df_forecast['dod'].mean()
mean_mom = df_forecast['mom'].mean(skipna=True)
mean_wow = df_forecast['wow'].mean(skipna=True)
total_2025 = df_forecast['forecast'].sum()

# === Wykres ===
ma = df_forecast['forecast'].rolling(ma_window, min_periods=1).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='📘 Historia'))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='🔮 Prognoza 2025'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))

fig.update_layout(
    title=f"Prognoza zamówień – {agg_type.lower()} agregacja (tylko 2025)",
    xaxis_title="Data",
    yaxis_title="Liczba zamówień",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.2)
)
st.plotly_chart(fig, use_container_width=True)

# === Analizy wzrostów ===
st.markdown("## 📈 Analiza wzrostów i trendów 2025")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Suma prognoz 2025", f"{total_2025:,.0f}")
col2.metric("📊 Średni wzrost D/D", f"{mean_dod:.2f}%")
col3.metric("📆 Średni wzrost W/W", f"{mean_wow:.2f}%" if not np.isnan(mean_wow) else "—")
col4.metric("🗓️ Średni wzrost M/M", f"{mean_mom:.2f}%" if not np.isnan(mean_mom) else "—")

# Dodatkowe statystyki
st.markdown("### 📊 Dodatkowe wskaźniki")
st.write(f"- Największy przyrost prognozy: **{df_forecast['forecast'].diff().max():,.0f}**")
st.write(f"- Największy spadek prognozy: **{df_forecast['forecast'].diff().min():,.0f}**")
st.write(f"- Odchylenie standardowe wzrostów dziennych: **{df_forecast['dod'].std():.2f}%**")
st.write(f"- Liczba punktów prognozy: **{len(df_forecast)}**")

# === Agregacja miesięczna dla przeglądu trendu ===
st.markdown("## 📅 Zestawienie miesięczne (2025)")
monthly = df_forecast.resample('M').sum()
monthly['MoM %'] = monthly['forecast'].pct_change() * 100
st.dataframe(monthly.style.format({'forecast': '{:,.0f}', 'MoM %': '{:.2f}%'}))

# === Wykres miesięczny ===
fig_m = go.Figure()
fig_m.add_trace(go.Bar(x=monthly.index.strftime("%Y-%m"), y=monthly['forecast'], name='Prognoza (miesięczna)'))
fig_m.update_layout(
    title="📊 Prognoza miesięczna – 2025",
    xaxis_title="Miesiąc",
    yaxis_title="Suma zamówień",
    template="plotly_white"
)
st.plotly_chart(fig_m, use_container_width=True)

# === Eksport ===
st.download_button("📥 Pobierz prognozę 2025 (CSV)", df_forecast.to_csv().encode(), file_name="forecast_2025_detailed.csv")
