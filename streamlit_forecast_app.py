# streamlit_forecast_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import holidays

st.set_page_config(page_title="Forecast Orders A+", layout="wide")
st.title("Forecast Orders")
st.markdown("Wgraj plik CSV/XLSX z kolumnami: data i orders oraz ustaw parametry.")

# Sidebar
st.sidebar.header("Ustawienia")
uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV/XLSX)", type=['csv','xlsx','xls'])
freq = st.sidebar.selectbox("Agregacja", ['Dzienna','Tygodniowa','Miesięczna'], index=0)
forecast_until = st.sidebar.date_input("Prognoza do daty", value=datetime(2025,12,31))
ma_window = st.sidebar.slider("Okno średniej kroczącej", 1, 60, 7)
rolling_window_days = st.sidebar.number_input("Długość okna walidacji (dni) dla rolling CV", min_value=7, max_value=120, value=30)
max_folds = st.sidebar.number_input("Liczba foldów rolling CV", min_value=1, max_value=6, value=3)

if uploaded_file is None:
    st.info("Prześlij plik z kolumnami: date i orders")
    st.stop()

# Wczytanie danych
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Błąd przy wczytywaniu pliku: {e}")
    st.stop()

# autodetect date and value columns
cols = df.columns.tolist()
date_col = next((c for c in cols if any(x in c.lower() for x in ['date','data','day'])), cols[0])
val_col = next((c for c in cols if any(x in c.lower() for x in ['order','orders','zam','ile','qty','quantity'])), [c for c in cols if c!=date_col][0])

data = df[[date_col, val_col]].copy()
data.columns = ['date','orders']
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)
data = data.set_index('date')

# Resampling
if freq=='Dzienna':
    ts = data['orders'].resample('D').sum()
    resample_rule = 'D'
elif freq=='Tygodniowa':
    ts = data['orders'].resample('W-MON').sum()
    resample_rule = 'W'
else:
    ts = data['orders'].resample('M').sum()
    resample_rule = 'M'

st.subheader('Podstawowe info o danych')
st.write('Zakres:', ts.index.min().date(), '—', ts.index.max().date())
st.write('Liczba punktów:', len(ts))
st.dataframe(ts.tail(10).rename('orders').to_frame())

# Polish holidays
years = list(range(ts.index.min().year, max(ts.index.max().year, pd.to_datetime(forecast_until).year)+1))
pl_holidays = holidays.CountryHoliday('PL', years=years)
holidays_df = pd.DataFrame({'ds': list(pl_holidays.keys()), 'holiday': list(pl_holidays.values())})

# Moving average
ma = ts.rolling(window=ma_window, min_periods=1).mean()

# Metrics
def mape(a, f):
    mask = np.array(a)!=0
    return np.mean(np.abs((np.array(a)[mask]-np.array(f)[mask])/np.array(a)[mask]))*100 if mask.sum()>0 else np.nan
def rmse(a,f): return np.sqrt(np.mean((np.array(a)-np.array(f))**2))
def mae(a,f): return np.mean(np.abs(np.array(a)-np.array(f)))

# Forecast functions
def fit_ets_forecast(train, periods, seasonal_periods=None, freq='D'):
    sp = seasonal_periods or (7 if freq=='D' else 12 if freq=='M' else 52)
    st_model = ExponentialSmoothing(train.fillna(0), trend='add', seasonal='add' if len(train)>2*sp else None, seasonal_periods=sp if len(train)>2*sp else None)
    fit = st_model.fit()
    preds = fit.forecast(periods)
    last = train.index[-1]
    idx = pd.date_range(start=last + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    preds.index = idx
    return preds

def fit_arima_forecast(train, periods, order=(1,1,1), freq='D'):
    model = ARIMA(train, order=order)
    fit = model.fit()
    preds = fit.forecast(periods)
    last = train.index[-1]
    idx = pd.date_range(start=last + pd.Timedelta(1, unit=freq), periods=periods, freq=freq)
    preds.index = idx
    return preds

# Rolling CV
st.subheader('Rolling cross-validation')
h = int(rolling_window_days)
n = len(ts)
min_train = max(30,h)
starts = [n - (max_folds-i)*h for i in range(max_folds) if n - (max_folds-i)*h > min_train] or [n-h-1]
st.write(f'Uruchamiam rolling CV z {len(starts)} foldami, każdy horizon = {h} punktów')

all_metrics = {'arima':[], 'ets':[]}
for idx, train_end in enumerate(starts):
    train = ts.iloc[:train_end]
    test = ts.iloc[train_end:train_end+h]
    if len(test)==0: continue
    st.write(f'Fold {idx+1}: trening do {train.index.max().date()}, test od {test.index.min().date()} ({len(test)} punktów)')
    # ARIMA
    try: a_pred = fit_arima_forecast(train, periods=len(test)); all_metrics['arima'].append((mape(test,a_pred), rmse(test,a_pred), mae(test,a_pred)))
    except: all_metrics['arima'].append((np.nan,np.nan,np.nan))
    # ETS
    try: e_pred = fit_ets_forecast(train, periods=len(test)); all_metrics['ets'].append((mape(test,e_pred), rmse(test,e_pred), mae(test,e_pred)))
    except: all_metrics['ets'].append((np.nan,np.nan,np.nan))

summary = {m:(np.nanmean([x[0] for x in vals]), np.nanmean([x[1] for x in vals]), np.nanmean([x[2] for x in vals])) for m,vals in all_metrics.items()}
met_df = pd.DataFrame(summary, index=['MAPE','RMSE','MAE']).T
st.subheader('Wyniki walidacji (rolling CV)')
st.dataframe(met_df.style.format("{:.2f}"))

# Best model
best_model = met_df['MAPE'].idxmin()
st.write('Najlepszy model wg MAPE:', best_model)

# Final forecast
st.subheader('Finalna prognoza')
last = ts.index.max()
end_date = pd.to_datetime(forecast_until)
periods = (end_date - last).days if freq=='Dzienna' else ((end_date - last).days)//7+1 if freq=='Tygodniowa' else (end_date.year - last.year)*12 + (end_date.month - last.month)
if periods>0:
    if best_model=='arima': final_pred = fit_arima_forecast(ts, periods)
    else: final_pred = fit_ets_forecast(ts, periods)
    full = pd.concat([ts, final_pred])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='History'))
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'MA ({ma_window})'))
    fig.add_trace(go.Scatter(x=final_pred.index, y=final_pred.values, mode='lines', name=f'Forecast ({best_model})'))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button('Pobierz prognozę (CSV)', data=final_pred.rename('forecast').to_csv().encode(), file_name='forecast.csv')
else:
    st.warning("Data prognozy jest przed ostatnią obserwacją. Wybierz późniejszą datę.")
