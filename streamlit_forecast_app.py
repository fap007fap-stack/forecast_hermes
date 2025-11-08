import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="📈 Advanced Forecast", layout="wide")
st.title("🛒 Prognoza zamówień eCommerce z analizą YoY, scenariuszami i szczytami")

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
else:
    ts = data.set_index('date')['orders'].resample('MS').sum()
    freq_rule = 'MS'

# === Skumulowane dane historyczne ===
ts_cum = ts.cumsum()
ma = ts.rolling(ma_window, min_periods=1).mean().cumsum()

# === Dane poprzedniego roku ===
prev_year = ts[ts.index.year == ts.index.max().year - 1]
curr_year = ts[ts.index.year == ts.index.max().year]
prev_year_sum = prev_year.sum()
curr_year_sum = curr_year.sum() if len(curr_year)>0 else np.nan
yoy_growth = ((curr_year_sum - prev_year_sum) / prev_year_sum * 100) if prev_year_sum>0 else np.nan

# === Forecast i scenariusze ===
forecast_horizon = (datetime(2025,12,31) - ts.index.max()).days if freq=='Dzienna' else \
                   (52 - ts.index.max().isocalendar()[1] if freq=='W' else 12 - ts.index.max().month)

# Sezonowość
if freq=='Dzienna':
    seasonal_periods = 365 if len(ts)>=730 else 7
elif freq=='W':
    seasonal_periods = 52 if len(ts)>=104 else 4
else:
    seasonal_periods = 12 if len(ts)>=24 else 3

try:
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    forecast_base = fit.forecast(forecast_horizon)
    forecast_base.index = pd.date_range(ts.index.max() + pd.Timedelta(1, unit=freq_rule),
                                        periods=forecast_horizon, freq=freq_rule)
    
    # Scenariusze prognozy
    forecast_cum_base = pd.Series(np.cumsum(forecast_base.values) + ts_cum.iloc[-1], index=forecast_base.index)
    forecast_cum_opt = forecast_cum_base * 1.1  # +10% optymistyczny
    forecast_cum_pess = forecast_cum_base * 0.9  # -10% pesymistyczny
    
except Exception as e:
    st.error(f"Błąd przy dopasowaniu modelu: {e}")
    st.stop()

# === Wizualizacja skumulowana z prognozami ===
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='📘 Historyczne'))
fig.add_trace(go.Scatter(x=forecast_cum_base.index, y=forecast_cum_base.values, mode='lines', name='🔮 Prognoza'))
fig.add_trace(go.Scatter(x=forecast_cum_opt.index, y=forecast_cum_opt.values, mode='lines', name='🔮 Optymistyczny'))
fig.add_trace(go.Scatter(x=forecast_cum_pess.index, y=forecast_cum_pess.values, mode='lines', name='🔮 Pesymistyczny'))
fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'Średnia krocząca ({ma_window})'))
fig.update_layout(title="Skumulowana prognoza zamówień",
                  xaxis_title="Data", yaxis_title="Skumulowana liczba zamówień",
                  template="plotly_white", legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)

# === Wskaźniki kluczowe ===
st.markdown("## 📊 Kluczowe wskaźniki")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prognoza całkowita 2025", f"{forecast_cum_base.iloc[-1]:,.0f}")
col2.metric("Wzrost YoY (rok do roku)", f"{yoy_growth:.2f}%" if not np.isnan(yoy_growth) else "Brak danych")
col3.metric("Średni dzienny wzrost", f"{ts.diff().mean():,.2f}")
col4.metric("Typ sezonowości", f"{seasonal_periods} jednostek")

# === Interaktywny wykres procentowych wzrostów ===
st.markdown("### 📈 Procentowe wzrosty")
daily_change = ts.pct_change() * 100
weekly_change = ts.resample('W-MON').sum().pct_change() * 100
monthly_change = ts.resample('MS').sum().pct_change() * 100
yoy_daily = ts.diff(365) / ts.shift(365) * 100 if len(ts)>365 else pd.Series(dtype=float)

fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=daily_change.index, y=daily_change, mode='lines', name='Dzienny [%]'))
fig_growth.add_trace(go.Scatter(x=weekly_change.index, y=weekly_change.reindex(daily_change.index, method='ffill'), mode='lines', name='Tygodniowy [%]'))
fig_growth.add_trace(go.Scatter(x=monthly_change.index, y=monthly_change.reindex(daily_change.index, method='ffill'), mode='lines', name='Miesięczny [%]'))
fig_growth.add_trace(go.Scatter(x=yoy_daily.index, y=yoy_daily.reindex(daily_change.index, method='ffill'), mode='lines', name='YoY Dzienny [%]'))
fig_growth.update_layout(title="Procentowe wzrosty dzienne/tygodniowe/miesięczne/YoY", xaxis_title="Data", yaxis_title="[%]", template="plotly_white")
st.plotly_chart(fig_growth, use_container_width=True)

# === Analiza szczytów ===
st.markdown("### 📊 Analiza szczytów")
st.write("Top 5 dni, tygodni i miesięcy w historii oraz prognozie:")

top_days = ts.sort_values(ascending=False).head(5)
top_weeks = ts.resample('W-MON').sum().sort_values(ascending=False).head(5)
top_months = ts.resample('MS').sum().sort_values(ascending=False).head(5)
top_forecast = forecast_base.sort_values(ascending=False).head(5)

st.write("📅 **Top 5 dni historycznych:**")
st.table(top_days)
st.write("📅 **Top 5 tygodni historycznych:**")
st.table(top_weeks)
st.write("📅 **Top 5 miesięcy historycznych:**")
st.table(top_months)
st.write("📅 **Top 5 dni prognozy 2025:**")
st.table(top_forecast)

# === Pobranie prognozy ===
st.download_button("📥 Pobierz prognozę 2025 (CSV)", forecast_cum_base.rename('forecast').to_csv().encode(), file_name="forecast_2025_cumulative.csv")
