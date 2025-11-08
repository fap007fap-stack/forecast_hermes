import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import holidays

st.set_page_config(page_title="Forecast Orders A+", layout="wide")
st.title("📈 Prognoza zamówień (z sumowaniem kumulacyjnym)")

# === Wczytanie danych ===
uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV/XLSX)", type=['csv','xlsx','xls'])
if uploaded_file is None:
    st.info("📁 Prześlij plik z kolumnami: data i wartość (orders/zamówienia)")
    st.stop()

if uploaded_file.name.endswith('.csv'):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Podgląd danych")
st.write(df.head())

# === Wybór kolumn ===
st.sidebar.header("Ustawienia kolumn")
date_col = st.sidebar.selectbox("Kolumna z datą", df.columns)
val_col = st.sidebar.selectbox("Kolumna z wartością (orders)", [c for c in df.columns if c != date_col])

# === Parametry ===
freq = st.sidebar.selectbox("Agregacja", ['Dzienna', 'Tygodniowa', 'Miesięczna'], index=0)
forecast_until = st.sidebar.date_input("Prognoza do daty", value=datetime(2025, 12, 31))
ma_window = st.sidebar.slider("Okno średniej kroczącej", 1, 60, 7)
rolling_window_days = st.sidebar.number_input("Długość okna walidacji (dni)", min_value=7, max_value=120, value=30)
max_folds = st.sidebar.number_input("Liczba foldów rolling CV", min_value=1, max_value=6, value=3)

# === Przygotowanie danych ===
data = df[[date_col, val_col]].copy()
data.columns = ['date', 'orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

# agregacja dzienna
if freq == 'Dzienna':
    ts = data.set_index('date')['orders'].resample('D').sum()
    freq_rule = 'D'
elif freq == 'Tygodniowa':
    ts = data.set_index('date')['orders'].resample('W-MON').sum()
    freq_rule = 'W'
else:
    ts = data.set_index('date')['orders'].resample('M').sum()
    freq_rule = 'M'

# === Kumulacja (sumowanie narastająco) ===
ts_cum = ts.cumsum()

st.subheader("📊 Dane po przetworzeniu")
st.write("Zakres:", ts_cum.index.min().date(), "—", ts_cum.index.max().date())
st.write("Liczba punktów:", len(ts_cum))
st.dataframe(ts_cum.tail(10).rename('orders').to_frame())

# === Średnia krocząca ===
ma = ts_cum.rolling(window=ma_window, min_periods=1).mean()

# === Miary błędu ===
def mape(a, f):
    mask = np.array(a) != 0
    return np.mean(np.abs((np.array(a)[mask] - np.array(f)[mask]) / np.array(a)[mask])) * 100 if mask.sum() > 0 else np.nan
def rmse(a, f): return np.sqrt(np.mean((np.array(a) - np.array(f)) ** 2))
def mae(a, f): return np.mean(np.abs(np.array(a) - np.array(f)))

# === Modele ===
def fit_ets_forecast(train, periods, seasonal_periods=None, freq='D'):
    sp = seasonal_periods or (7 if freq == 'D' else 12 if freq == 'M' else 52)
    model = ExponentialSmoothing(train, trend='add', seasonal='add' if len(train) > 2 * sp else None, seasonal_periods=sp if len(train) > 2 * sp else None)
    fit = model.fit()
    pred = fit.forecast(periods)
    pred.index = pd.date_range(train.index[-1] + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    return pred

def fit_arima_forecast(train, periods, order=(1, 1, 1), freq='D'):
    model = ARIMA(train, order=order)
    fit = model.fit()
    pred = fit.forecast(periods)
    pred.index = pd.date_range(train.index[-1] + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    return pred

# === Rolling CV ===
st.subheader("🔁 Rolling cross-validation")
h = int(rolling_window_days)
n = len(ts_cum)
min_train = max(30, h)
starts = [n - (max_folds - i) * h for i in range(max_folds) if n - (max_folds - i) * h > min_train] or [n - h - 1]

all_metrics = {'arima': [], 'ets': []}
for i, train_end in enumerate(starts):
    train = ts_cum.iloc[:train_end]
    test = ts_cum.iloc[train_end:train_end + h]
    if len(test) == 0: continue
    st.write(f"Fold {i+1}: trening do {train.index.max().date()}, test od {test.index.min().date()} ({len(test)} dni)")

    try:
        a_pred = fit_arima_forecast(train, len(test))
        all_metrics['arima'].append((mape(test, a_pred), rmse(test, a_pred), mae(test, a_pred)))
    except:
        all_metrics['arima'].append((np.nan, np.nan, np.nan))

    try:
        e_pred = fit_ets_forecast(train, len(test))
        all_metrics['ets'].append((mape(test, e_pred), rmse(test, e_pred), mae(test, e_pred)))
    except:
        all_metrics['ets'].append((np.nan, np.nan, np.nan))

summary = {m: (np.nanmean([x[0] for x in vals]),
               np.nanmean([x[1] for x in vals]),
               np.nanmean([x[2] for x in vals])) for m, vals in all_metrics.items()}
met_df = pd.DataFrame(summary, index=['MAPE', 'RMSE', 'MAE']).T
st.dataframe(met_df.style.format("{:.2f}"))

best_model = met_df['MAPE'].idxmin()
st.write(f"🏆 Najlepszy model: **{best_model}**")

# === Prognoza finalna ===
st.subheader("📈 Prognoza końcowa")
last = ts_cum.index.max()
end_date = pd.to_datetime(forecast_until)
periods = (end_date - last).days if freq == 'Dzienna' else ((end_date - last).days) // 7 + 1 if freq == 'Tygodniowa' else (end_date.year - last.year) * 12 + (end_date.month - last.month)

if periods > 0:
    final_pred = fit_arima_forecast(ts_cum, periods) if best_model == 'arima' else fit_ets_forecast(ts_cum, periods)
    full = pd.concat([ts_cum, final_pred])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_cum.index, y=ts_cum.values, mode='lines', name='Historyczne (kumulowane)'))
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'MA ({ma_window})'))
    fig.add_trace(go.Scatter(x=final_pred.index, y=final_pred.values, mode='lines', name=f'Prognoza ({best_model})'))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button('📥 Pobierz prognozę (CSV)',
                       data=final_pred.rename('forecast').to_csv().encode(),
                       file_name='forecast_cumulative.csv')
else:
    st.warning("Data prognozy jest przed ostatnią obserwacją.")
