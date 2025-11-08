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

# === Modelowanie prognozy bez sezonowości z poprzedniego roku ===
st.subheader("📈 Modelowanie prognozy")
try:
    # automatyczne dopasowanie trendu i sezonowości
    seasonal_periods = 7 if len(ts) < 100 else 30 if len(ts) < 365 else 365
    model = ExponentialSmoothing(ts_cum, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)

    last_date = ts_cum.index.max()
    forecast_horizon = (datetime(2025, 12, 31) - last_date).days if freq == 'Dzienna' else \
                       ((52 if freq == 'W' else 12) - ((last_date.month-1) if freq == 'MS' else 0))
    forecast = fit.forecast(forecast_horizon)
    forecast.index = pd.date_range(last_date + pd.Timedelta(1, unit=freq_rule), periods=forecast_horizon, freq=freq_rule)

except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

full = pd.concat([ts_cum, forecast])
ma = ts_cum.rolling(ma_window, min_periods=1).mean()

# === Podsumowanie całego roku 2025 ===
forecast_2025_sum = forecast.sum()

# === Wskaźniki wzrostu ===
daily_diff = ts.diff()
weekly_diff = ts.resample('W-MON').sum().diff()
monthly_diff = ts.resample('MS').sum().diff()

# === Wizualizacja ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='📘 Historyczne (kumulowane)'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name=f'🔮 Prognoza'))

fig.update_layout(
    title=f"Prognoza skumulowanych zamówień (bez sezonowości z poprzedniego roku)",
    xaxis_title="Data",
    yaxis_title="Skumulowana liczba zamówień",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# === Dodatkowe statystyki ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prognoza całkowita 2025", f"{forecast_2025_sum:,.0f}")
col2.metric("Średni dzienny wzrost", f"{daily_diff.mean():,.2f}")
col3.metric("Średni tygodniowy wzrost", f"{weekly_diff.mean():,.2f}")
col4.metric("Średni miesięczny wzrost", f"{monthly_diff.mean():,.2f}")

st.download_button("📥 Pobierz prognozę (CSV)", forecast.rename('forecast').to_csv().encode(), file_name="forecast_2025.csv")
