import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="📈 Forecast Orders – Advanced Daily Model", layout="wide")
st.title("🛒 Prognoza dzienna zamówień eCommerce z zaawansowanymi średnimi kroczącymi i scenariuszami")

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

# === Parametry dodatkowe ===
ma_window = st.sidebar.slider("Średnia krocząca (dni)", 3, 30, 7)
season_input = st.sidebar.number_input("Okres sezonowości (dni)", min_value=2, max_value=730, value=365)
opt_change = st.sidebar.slider("Zmiana dla scenariuszy [%]", -50, 50, 10)

# === Przygotowanie danych ===
data = df[[date_col, val_col]].copy()
data.columns = ['date', 'orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

# === Agregacja dzienna ===
ts = data.set_index('date')['orders'].resample('D').sum()
ts_cum = ts.cumsum()

# === Średnie kroczące historyczne ===
ma_short = ts.rolling(7, min_periods=1).mean().cumsum()
ma_mid = ts.rolling(30, min_periods=1).mean().cumsum()
ma_long = ts.rolling(90, min_periods=1).mean().cumsum()
ema_30 = ts.ewm(span=30, adjust=False).mean().cumsum()

# === Dane poprzedniego roku i wzrost YoY ===
prev_year = ts[ts.index.year == ts.index.max().year - 1]
prev_year_sum = prev_year.sum()
curr_year_sum = ts[ts.index.year == ts.index.max().year].sum() if any(ts.index.year==ts.index.max().year) else np.nan
yoy_growth = ((curr_year_sum - prev_year_sum)/prev_year_sum*100) if prev_year_sum>0 else np.nan

# === Forecast ===
forecast_horizon = (datetime(2025,12,31) - ts.index.max()).days
try:
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=season_input)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(forecast_horizon)
    forecast.index = pd.date_range(ts.index.max() + pd.Timedelta(1, unit='D'), periods=forecast_horizon, freq='D')

    # Skumulowane prognozy
    forecast_cum = pd.Series(np.cumsum(forecast.values) + ts_cum.iloc[-1], index=forecast.index)
    forecast_cum_opt = forecast_cum * (1 + opt_change/100)
    forecast_cum_pess = forecast_cum * (1 - opt_change/100)

    # Średnie kroczące + forecast (poprawnie łączone)
    ma_short_full = pd.concat([ma_short, forecast_cum.rolling(7, min_periods=1).mean()])
    ma_mid_full = pd.concat([ma_mid, forecast_cum.rolling(30, min_periods=1).mean()])
    ma_long_full = pd.concat([ma_long, forecast_cum.rolling(90, min_periods=1).mean()])
    ema_30_full = pd.concat([ema_30, forecast_cum.ewm(span=30, adjust=False).mean()])

except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

# === Wizualizacja ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='📘 Historyczne'))
fig.add_trace(go.Scatter(x=forecast_cum.index, y=forecast_cum.values, mode='lines', name='🔮 Prognoza'))
fig.add_trace(go.Scatter(x=forecast_cum_opt.index, y=forecast_cum_opt.values, mode='lines', name='🔮 Optymistyczny'))
fig.add_trace(go.Scatter(x=forecast_cum_pess.index, y=forecast_cum_pess.values, mode='lines', name='🔮 Pesymistyczny'))

# === Średnie kroczące historyczne + forecast (przerywane, delikatny kolor) ===
fig.add_trace(go.Scatter(x=ma_short_full.index, y=ma_short_full.values, mode='lines', name='MA 7 dni',
                         line=dict(dash='dot', color='lightblue')))
fig.add_trace(go.Scatter(x=ma_mid_full.index, y=ma_mid_full.values, mode='lines', name='MA 30 dni',
                         line=dict(dash='dot', color='lightgreen')))
fig.add_trace(go.Scatter(x=ma_long_full.index, y=ma_long_full.values, mode='lines', name='MA 90 dni',
                         line=dict(dash='dot', color='lightcoral')))
fig.add_trace(go.Scatter(x=ema_30_full.index, y=ema_30_full.values, mode='lines', name='EMA 30 dni',
                         line=dict(dash='dot', color='orange')))

fig.update_layout(title="Skumulowana prognoza dzienna zamówień z zaawansowanymi średnimi kroczącymi",
                  xaxis_title="Data", yaxis_title="Skumulowana liczba zamówień",
                  template="plotly_white", legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)

# === Wskaźniki kluczowe ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3 = st.columns(3)
col1.metric("Prognoza całkowita 2025", f"{forecast_cum.iloc[-1]:,.0f}")
col2.metric("Wzrost YoY (rok do roku)", f"{yoy_growth:.2f}%" if not np.isnan(yoy_growth) else "Brak danych")
col3.metric("Średni dzienny wzrost", f"{ts.diff().mean():,.2f}")

# === Pobranie prognozy ===
st.download_button("📥 Pobierz prognozę (CSV)", forecast_cum.rename('forecast').to_csv().encode(), file_name="forecast_2025_cumulative.csv")
