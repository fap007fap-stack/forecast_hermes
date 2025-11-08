import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import holidays

st.set_page_config(page_title="📈 Forecast Orders – Advanced Seasonal Model", layout="wide")
st.title("🛒 Prognoza zamówień eCommerce (z sezonowością roczną i świąteczną)")

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
freq = st.sidebar.selectbox("Agregacja", ['Dzienna', 'Tygodniowa'], index=0)
ma_window = st.sidebar.slider("Średnia krocząca (dni)", 3, 30, 7)

# === Przygotowanie danych ===
data = df[[date_col, val_col]].copy()
data.columns = ['date', 'orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

if freq == 'Dzienna':
    ts = data.set_index('date')['orders'].resample('D').sum()
    freq_rule = 'D'
    seasonal_periods = 365
else:
    ts = data.set_index('date')['orders'].resample('W-MON').sum()
    freq_rule = 'W'
    seasonal_periods = 52

ts_cum = ts.cumsum()

st.subheader("📅 Zakres danych")
st.write(f"Od **{ts_cum.index.min().date()}** do **{ts_cum.index.max().date()}**, liczba punktów: **{len(ts_cum)}**")

# === Dekompzycja sezonowości ===
with st.expander("🔍 Analiza sezonowości"):
    try:
        result = seasonal_decompose(ts, model='additive', period=seasonal_periods)
        fig_dec = go.Figure()
        fig_dec.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal.values, mode='lines', name='Sezonowość'))
        fig_dec.update_layout(title="Komponent sezonowy (średni roczny wzorzec)")
        st.plotly_chart(fig_dec, use_container_width=True)
    except Exception as e:
        st.warning(f"Nie udało się przeprowadzić dekompozycji sezonowości: {e}")

# === Model z roczną sezonowością ===
st.subheader("📈 Modelowanie prognozy")
try:
    model = ExponentialSmoothing(ts_cum, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    last_date = ts_cum.index.max()
    forecast_horizon = (datetime(2025, 12, 31) - last_date).days if freq == 'Dzienna' else 52
    forecast = fit.forecast(forecast_horizon)
    forecast.index = pd.date_range(last_date + pd.Timedelta(1, unit=freq_rule), periods=forecast_horizon, freq=freq_rule)
except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

full = pd.concat([ts_cum, forecast])
ma = ts_cum.rolling(ma_window, min_periods=1).mean()

# === Analiza YoY ===
hist_2024 = ts_cum.loc[ts_cum.index.year == 2024].iloc[-1] if any(ts_cum.index.year == 2024) else np.nan
forecast_2025 = forecast.iloc[-1]
yoy_growth = ((forecast_2025 - hist_2024) / hist_2024 * 100) if not np.isnan(hist_2024) else np.nan

# === Wizualizacja ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='📘 Historyczne (kumulowane)'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name='🔮 Prognoza 2025 (z roczną sezonowością)'))

fig.update_layout(
    title="Prognoza skumulowanych zamówień z roczną sezonowością (e-commerce)",
    xaxis_title="Data",
    yaxis_title="Skumulowana liczba zamówień",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# === Dodatkowe statystyki ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prognoza na koniec 2025", f"{forecast_2025:,.0f}")
col2.metric("Wzrost YoY (2025 vs 2024)", f"{yoy_growth:.2f}%" if not np.isnan(yoy_growth) else "Brak danych 2024")
col3.metric("Średni dzienny wzrost", f"{ts_cum.diff().mean():,.2f}")
col4.metric("Okno MA", f"{ma_window} dni")

st.markdown("### 📅 Dodatkowe dane sezonowe")
st.write(f"- Model uwzględnia **roczny cykl 365 dni**, co pozwala przewidzieć wzrost w okresie świątecznym 🎅")
st.write(f"- Wykryto trend: **{fit.params['smoothing_trend']:.4f}**, wzmocnienie sezonowości: **{fit.params['smoothing_seasonal']:.4f}**")
st.write(f"- Prognoza obejmuje okres: **{last_date.date()} → 2025-12-31**")

st.download_button("📥 Pobierz prognozę (CSV)", forecast.rename('forecast').to_csv().encode(), file_name="forecast_2025_seasonal.csv")
