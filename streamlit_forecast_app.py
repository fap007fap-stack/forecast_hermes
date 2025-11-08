import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="📈 Forecast Orders – Smart Model", layout="wide")
st.title("🛒 Prognoza zamówień eCommerce (bez sezonowości z poprzedniego roku)")

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

# === Parametry ===
freq = st.sidebar.selectbox("Agregacja", ['Dzienna', 'Tygodniowa', 'Miesięczna'], index=0)
ma_window = st.sidebar.slider("Średnia krocząca (dni)", 3, 30, 7)

# === Przygotowanie danych ===
data = df[[date_col, val_col]].copy()
data.columns = ['date', 'orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

# === Agregacja danych ===
if freq == 'Dzienna':
    ts = data.set_index('date')['orders'].resample('D').sum()
    freq_rule = 'D'
elif freq == 'Tygodniowa':
    ts = data.set_index('date')['orders'].resample('W-MON').sum()
    freq_rule = 'W'
else:  # Miesięczna
    ts = data.set_index('date')['orders'].resample('MS').sum()
    freq_rule = 'MS'

ts_cum = ts.cumsum()

st.subheader("📅 Zakres danych")
st.write(f"Od **{ts_cum.index.min().date()}** do **{ts_cum.index.max().date()}**, liczba punktów: **{len(ts_cum)}**")

# === Modelowanie prognozy ===
st.subheader("📈 Modelowanie prognozy")
try:
    # automatyczne dopasowanie sezonowości tylko jeśli mamy minimum 2 cykle
    if freq == 'Dzienna':
        seasonal_periods = 365 if len(ts) >= 730 else None
    elif freq == 'Tygodniowa':
        seasonal_periods = 52 if len(ts) >= 104 else None
    else:  # Miesięczna
        seasonal_periods = 12 if len(ts) >= 24 else None

    model = ExponentialSmoothing(ts, trend='add', seasonal='add' if seasonal_periods else None,
                                 seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)

    last_date = ts.index.max()
    if freq == 'Dzienna':
        forecast_horizon = (datetime(2025,12,31) - last_date).days
    elif freq == 'Tygodniowa':
        forecast_horizon = 52 - last_date.isocalendar()[1]
    else:  # Miesięczna
        forecast_horizon = 12 - last_date.month

    forecast = fit.forecast(forecast_horizon)
    forecast.index = pd.date_range(last_date + pd.Timedelta(1, unit=freq_rule), periods=forecast_horizon, freq=freq_rule)

except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

# === Sumaryczna prognoza 2025 ===
forecast_2025_sum = forecast.sum()

# === Średnie wzrosty ===
daily_diff = ts.diff()
weekly_diff = ts.resample('W-MON').sum().diff()
monthly_diff = ts.resample('MS').sum().diff()

# === Średnia krocząca ===
ma = ts.rolling(ma_window, min_periods=1).mean()

# === Wizualizacja ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='📘 Historyczne'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name=f'🔮 Prognoza'))

fig.update_layout(
    title="Prognoza zamówień eCommerce (bez sezonowości z poprzedniego roku)",
    xaxis_title="Data",
    yaxis_title="Liczba zamówień",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# === Dodatkowe wskaźniki ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prognoza całkowita 2025", f"{forecast_2025_sum:,.0f}")
col2.metric("Średni dzienny wzrost", f"{daily_diff.mean():,.2f}")
col3.metric("Średni tygodniowy wzrost", f"{weekly_diff.mean():,.2f}")
col4.metric("Średni miesięczny wzrost", f"{monthly_diff.mean():,.2f}")

# === Pobranie prognozy ===
st.download_button("📥 Pobierz prognozę (CSV)", forecast.rename('forecast').to_csv().encode(), file_name="forecast_2025.csv")
