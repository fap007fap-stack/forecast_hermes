import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="📈 Forecast Orders – Advanced YoY Model", layout="wide")
st.title("🛒 Prognoza zamówień eCommerce z uwzględnieniem danych z poprzedniego roku")

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

# === Agregacja ===
if freq == 'Dzienna':
    ts = data.set_index('date')['orders'].resample('D').sum()
    freq_rule = 'D'
elif freq == 'Tygodniowa':
    ts = data.set_index('date')['orders'].resample('W-MON').sum()
    freq_rule = 'W'
else:  # Miesięczna
    ts = data.set_index('date')['orders'].resample('MS').sum()
    freq_rule = 'MS'

# === Dane poprzedniego roku ===
prev_year = ts[ts.index.year == ts.index.max().year - 1]
curr_year = ts[ts.index.year == ts.index.max().year]

# === Obliczenie wzrostu YoY z poprzedniego roku ===
prev_year_sum = prev_year.sum()
curr_year_sum = curr_year.sum() if len(curr_year)>0 else np.nan
yoy_growth = ((curr_year_sum - prev_year_sum) / prev_year_sum * 100) if prev_year_sum>0 else np.nan

# Wzrost dzienny / tygodniowy / miesięczny YoY
daily_yoy = ts.diff(365) if freq=='Dzienna' and len(ts)>=366 else None
weekly_yoy = ts.resample('W-MON').sum().diff(52) if freq=='W' and len(ts)>=104 else None
monthly_yoy = ts.resample('MS').sum().diff(12) if freq=='MS' and len(ts)>=24 else None

# === Skumulowane dane historyczne ===
ts_cum = ts.cumsum()
ma = ts.rolling(ma_window, min_periods=1).mean().cumsum()

# === Forecast z dopasowaniem do poprzedniego roku ===
forecast_horizon = (datetime(2025,12,31) - ts.index.max()).days if freq=='Dzienna' else \
                   (52 - ts.index.max().isocalendar()[1] if freq=='W' else 12 - ts.index.max().month)

# Sprawdzenie sezonowości
if freq=='Dzienna':
    seasonal_periods = 365 if len(ts)>=730 else 7
elif freq=='W':
    seasonal_periods = 52 if len(ts)>=104 else 4
else:
    seasonal_periods = 12 if len(ts)>=24 else 3

try:
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(forecast_horizon)
    forecast.index = pd.date_range(ts.index.max() + pd.Timedelta(1, unit=freq_rule),
                                   periods=forecast_horizon, freq=freq_rule)
    # Skumulowanie prognozy
    forecast_cum = pd.Series(np.cumsum(forecast.values) + ts_cum.iloc[-1], index=forecast.index)
except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

# === Skumulowane dane łącznie z forecastem ===
full_cum = pd.concat([ts_cum, forecast_cum])

# === Wizualizacja ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='📘 Historyczne (kumulowane)'))
fig.add_trace(go.Scatter(x=forecast_cum.index, y=forecast_cum.values, mode='lines', name='🔮 Prognoza (kumulowana)'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))

fig.update_layout(
    title=f"Skumulowana prognoza zamówień z uwzględnieniem danych z poprzedniego roku",
    xaxis_title="Data",
    yaxis_title="Skumulowana liczba zamówień",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.25)
)
st.plotly_chart(fig, use_container_width=True)

# === Dodatkowe wskaźniki ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prognoza całkowita 2025", f"{forecast_cum.iloc[-1]:,.0f}")
col2.metric("Wzrost YoY (rok do roku)", f"{yoy_growth:.2f}%" if not np.isnan(yoy_growth) else "Brak danych")
col3.metric("Średni dzienny wzrost", f"{ts.diff().mean():,.2f}")
col4.metric("Typ sezonowości", f"{seasonal_periods} jednostek")

# === Dodatkowe wzrosty względem poprzedniego roku ===
st.markdown("### 📈 Wzrosty YoY")
if daily_yoy is not None:
    st.write("Dzienny YoY (różnica z tym samym dniem w poprzednim roku)")
    st.line_chart(daily_yoy.fillna(0))
if weekly_yoy is not None:
    st.write("Tygodniowy YoY")
    st.line_chart(weekly_yoy.fillna(0))
if monthly_yoy is not None:
    st.write("Miesięczny YoY")
    st.line_chart(monthly_yoy.fillna(0))

# === Pobranie prognozy ===
st.download_button("📥 Pobierz prognozę (CSV)", forecast_cum.rename('forecast').to_csv().encode(), file_name="forecast_2025_cumulative.csv")
