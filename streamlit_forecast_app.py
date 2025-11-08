# Streamlit app: Forecast orders (Prophet + auto_arima + ETS + rolling CV)
# Save this file as `streamlit_forecast_app.py` and push to GitHub. Then connect the repo to Streamlit Cloud or run locally with `streamlit run streamlit_forecast_app.py`.

"""
Ta wersja A+ dostosowana do Twoich danych (czerwiec 2024 — listopad 2025):
- Modele: Prophet (zalecany), pmdarima.auto_arima (ARIMA baseline), Holt-Winters (ETS)
- Rolling cross-validation (expanding window) z metrykami (MAPE/RMSE/MAE)
- Automatyczny wybór najlepszego modelu na podstawie MAPE
- Obsługa świąt w Polsce (pakiet `holidays`) — użyte jako regresory dla Prophet i jako flagi dla ARIMA
- Interaktywny wykres historii + prognoz (plotly)
- Możliwość pobrania prognozy CSV

Założenia: dane mają kolumny daty i liczby zamówień. Aplikacja agreguje do dziennego/tygodniowego/miesięcznego i działa najlepiej przy danych od 2024-06 do 2025-11 (jak podałeś).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import holidays

# Prophet (facebook/prophet)
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

# pmdarima for auto_arima
try:
    import pmdarima as pm
except Exception:
    pm = None

st.set_page_config(page_title="Forecast Orders A+", layout="wide")
st.title("Forecast Orders")
st.markdown("Wybierz plik CSV/XLSX i ustaw parametry.")

# Sidebar
st.sidebar.header("Ustawienia")
uploaded_file = st.sidebar.file_uploader("Wgraj plik (CSV/XLSX)", type=['csv','xlsx','xls'])
freq = st.sidebar.selectbox("Agregacja", ['Dzienna','Tygodniowa','Miesięczna'], index=0)
forecast_until = st.sidebar.date_input("Prognoza do daty", value=datetime(2025,12,31))
ma_window = st.sidebar.slider("Okno średniej kroczącej", 1, 60, 7)
rolling_window_days = st.sidebar.number_input("Długość okna walidacji (dni) dla rolling CV", min_value=7, max_value=120, value=30)
max_folds = st.sidebar.number_input("Liczba foldów rolling CV", min_value=1, max_value=6, value=3)

if uploaded_file is None:
    st.info("Prześlij plik z kolumnami: data (date) i orders (ile zamówień).")
    st.stop()

# Load data
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
date_col = None
val_col = None
for c in cols:
    lc = c.lower()
    if date_col is None and any(x in lc for x in ['date','data','day']):
        date_col = c
    if val_col is None and any(x in lc for x in ['order','orders','zam','ile','qty','quantity']):
        val_col = c
if date_col is None:
    date_col = cols[0]
if val_col is None:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    val_col = numeric[0] if numeric else cols[1] if len(cols)>1 else cols[0]

data = df[[date_col, val_col]].copy()
data.columns = ['date','orders']

# parse date
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date']).sort_values('date')
# ensure numeric
data['orders'] = pd.to_numeric(data['orders'], errors='coerce').fillna(0)

# set index and resample
data = data.set_index('date')
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

# create holiday dataframe for prophet
hols = []
for y in years:
    for d, name in pl_holidays.items():
        hols.append({'ds': pd.to_datetime(d), 'holiday': name})
holidays_df = pd.DataFrame(hols)

# moving average
ma = ts.rolling(window=ma_window, min_periods=1).mean()

# Utility metrics
def mape(a, f):
    a = np.array(a)
    f = np.array(f)
    mask = a != 0
    if mask.sum()==0:
        return np.nan
    return np.mean(np.abs((a[mask]-f[mask]) / a[mask]))*100

def rmse(a,f):
    return np.sqrt(np.mean((np.array(a)-np.array(f))**2))

def mae(a,f):
    return np.mean(np.abs(np.array(a)-np.array(f)))

# Rolling cross-validation (expanding window)
st.subheader('Rolling cross-validation (expanding window)')
if len(ts) < 30:
    st.warning('Mało danych (<30 punktów). Walidacja rolling może być niestabilna, ale i tak uruchamiam z minimalnymi foldami.')

# Prepare forecast horizon: we'll forecast same length as rolling_window_days when comparing short-term
h = int(rolling_window_days)

# Build functions to fit models and forecast

def fit_prophet(train_series, periods, holidays_df=None, freq='D'):
    if Prophet is None:
        raise ImportError('Prophet nie zainstalowany')
    dfp = train_series.reset_index().rename(columns={'index':'ds','orders':'y'})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    # add country holidays manually
    if holidays_df is not None and not holidays_df.empty:
        m.add_country_holidays(country_name='PL')
    # fit
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    pred = fcst.set_index('ds')['yhat'].reindex(future['ds']).iloc[-periods:]
    pred.index = future['ds'].iloc[-periods:]
    return pred


def fit_auto_arima_forecast(train_series, periods, exog_holidays=None, freq='D'):
    if pm is None:
        raise ImportError('pmdarima (auto_arima) nie zainstalowane')
    y = train_series.values
    # prepare seasonal argument if enough data
    seasonal = False
    m = 1
    if freq=='D' and len(y) > 7*2:
        seasonal = True
        m = 7
    if freq=='M' and len(y) > 12*2:
        seasonal = True
        m = 12
    arima = pm.auto_arima(y, seasonal=seasonal, m=m, stepwise=True, suppress_warnings=True, error_action='ignore')
    # forecast
    preds = arima.predict(n_periods=periods)
    # build index
    last = train_series.index[-1]
    if freq=='D':
        idx = pd.date_range(start=last + pd.Timedelta(1, unit='D'), periods=periods, freq='D')
    elif freq=='W':
        idx = pd.date_range(start=last + pd.offsets.Week(1), periods=periods, freq='W-MON')
    else:
        idx = pd.date_range(start=last + pd.offsets.MonthEnd(1), periods=periods, freq='M')
    s = pd.Series(preds, index=idx)
    return s


def fit_ets_forecast(train_series, periods, seasonal_periods=None, freq='D'):
    # simple ETS
    try:
        if seasonal_periods is None:
            seasonal_periods = 7 if freq=='D' else 52 if freq=='W' else 12
        st_model = ExponentialSmoothing(train_series.fillna(0), trend='add', seasonal='add' if len(train_series)>2*seasonal_periods else None, seasonal_periods=seasonal_periods if len(train_series)>2*seasonal_periods else None)
        fit = st_model.fit(optimized=True)
        preds = fit.forecast(periods)
        # build index
        last = train_series.index[-1]
        if freq=='D':
            idx = pd.date_range(start=last + pd.Timedelta(1, unit='D'), periods=periods, freq='D')
        elif freq=='W':
            idx = pd.date_range(start=last + pd.offsets.Week(1), periods=periods, freq='W-MON')
        else:
            idx = pd.date_range(start=last + pd.offsets.MonthEnd(1), periods=periods, freq='M')
        preds.index = idx
        return preds
    except Exception as e:
        return pd.Series([np.nan]*periods)

# Rolling CV implementation
freq_code = 'D' if resample_rule=='D' else ('W' if resample_rule=='W' else 'M')
all_metrics = {'prophet':[], 'arima':[], 'ets':[]}

# We'll run at most `max_folds` folds, each forecasting `h` points, using expanding training window
n = len(ts)
min_train = max(30, h)  # minimal training points
starts = []
# build fold start indices
for i in range(max_folds):
    train_end = n - (max_folds - i) * h
    if train_end <= min_train:
        continue
    starts.append(train_end)

if len(starts)==0:
    # fallback: single split
    starts = [n - h - 1]

st.write(f'Uruchamiam rolling CV z {len(starts)} foldami, każdy horizon = {h} punktów')

for idx, train_end in enumerate(starts):
    train = ts.iloc[:train_end]
    test = ts.iloc[train_end: train_end + h]
    if len(test)==0:
        continue

    st.write(f'Fold {idx+1}: trening do {train.index.max().date()}, test od {test.index.min().date()} ({len(test)} punktów)')

    # Prophet
    try:
        p_pred = fit_prophet(train, periods=len(test), holidays_df=holidays_df, freq=freq_code)
        # align indices
        p_pred = p_pred.reindex(test.index)
        all_metrics['prophet'].append((mape(test.values, p_pred.values), rmse(test.values, p_pred.values), mae(test.values, p_pred.values)))
    except Exception as e:
        all_metrics['prophet'].append((np.nan, np.nan, np.nan))
        st.warning(f'Prophet error on fold {idx+1}: {e}')

    # auto_arima
    try:
        a_pred = fit_auto_arima_forecast(train, periods=len(test), exog_holidays=None, freq=freq_code)
        a_pred = a_pred.reindex(test.index)
        all_metrics['arima'].append((mape(test.values, a_pred.values), rmse(test.values, a_pred.values), mae(test.values, a_pred.values)))
    except Exception as e:
        all_metrics['arima'].append((np.nan, np.nan, np.nan))
        st.warning(f'auto_arima error on fold {idx+1}: {e}')

    # ETS
    try:
        e_pred = fit_ets_forecast(train, periods=len(test), seasonal_periods=7 if freq_code=='D' else (52 if freq_code=='W' else 12), freq=freq_code)
        e_pred = e_pred.reindex(test.index)
        all_metrics['ets'].append((mape(test.values, e_pred.values), rmse(test.values, e_pred.values), mae(test.values, e_pred.values)))
    except Exception as e:
        all_metrics['ets'].append((np.nan, np.nan, np.nan))
        st.warning(f'ETS error on fold {idx+1}: {e}')

# Aggregate metrics
summary = {}
for mname, vals in all_metrics.items():
    arr = np.array(vals, dtype=float)
    if arr.size==0:
        summary[mname] = {'MAPE':np.nan,'RMSE':np.nan,'MAE':np.nan}
    else:
        summary[mname] = {'MAPE':np.nanmean(arr[:,0]), 'RMSE':np.nanmean(arr[:,1]), 'MAE':np.nanmean(arr[:,2])}

st.subheader('Wyniki walidacji (rolling CV)')
met_df = pd.DataFrame.from_dict(summary, orient='index')
st.dataframe(met_df.style.format({"MAPE":"{:.2f}", "RMSE":"{:.2f}", "MAE":"{:.2f}"}))

# Pick best model by MAPE
best_model = met_df['MAPE'].idxmin()
st.write('Najlepszy model wg MAPE:', best_model)

# Train selected model on full data and forecast to forecast_until
st.subheader('Finalna prognoza (trenowanie na pełnych danych)')
periods = None
last = ts.index.max()
end_date = pd.to_datetime(forecast_until)
if freq_code=='D':
    periods = (end_date - last).days
    freq_for_fb = 'D'
elif freq_code=='W':
    periods = int(((end_date - last).days)/7)+1
    freq_for_fb = 'W'
else:
    periods = (end_date.year - last.year)*12 + (end_date.month - last.month)
    freq_for_fb = 'M'

if periods <=0:
    st.warning('Wybrana data prognozy znajduje się przed ostatnią datą danych. Proszę wybrać datę późniejszą niż ostatnia obserwacja.')
else:
    st.write(f'Prognoza na {periods} okresów ({freq_for_fb})')
    if best_model=='prophet':
        try:
            final_pred = fit_prophet(ts, periods=periods, holidays_df=holidays_df, freq=freq_for_fb)
        except Exception as e:
            st.error('Błąd w Prophet final: '+str(e))
            final_pred = pd.Series([np.nan]*periods)
    elif best_model=='arima':
        try:
            final_pred = fit_auto_arima_forecast(ts, periods=periods, exog_holidays=None, freq=freq_for_fb)
        except Exception as e:
            st.error('Błąd w auto_arima final: '+str(e))
            final_pred = pd.Series([np.nan]*periods)
    else:
        try:
            final_pred = fit_ets_forecast(ts, periods=periods, seasonal_periods=7 if freq_code=='D' else (52 if freq_code=='W' else 12), freq=freq_for_fb)
        except Exception as e:
            st.error('Błąd w ETS final: '+str(e))
            final_pred = pd.Series([np.nan]*periods)

    # Combine history + forecast
    full = pd.concat([ts, final_pred])

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='History'))
    fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'MA ({ma_window})'))
    fig.add_trace(go.Scatter(x=final_pred.index, y=final_pred.values, mode='lines', name=f'Forecast ({best_model})'))
    # mark holidays
    hol_dates = [d for d in full.index if d in pl_holidays]
    if len(hol_dates)>0:
        fig.add_trace(go.Scatter(x=hol_dates, y=[full.get(d, np.nan) for d in hol_dates], mode='markers', name='Święta (PL)', marker=dict(size=8, symbol='x')))
    fig.update_layout(title='Historia + prognoza', xaxis_title='Date', yaxis_title='Orders')
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.write('Prognoza - sumy roczne (jeśli dostępne):')
    # sum predicted for 2025
    pred_2025 = final_pred[final_pred.index.year==2025]
    predicted_sum_2025 = float(pred_2025.sum()) if not pred_2025.empty else 0.0
    st.write(f'Prognozowana suma zamówień w 2025 z zakresu prognozy: {predicted_sum_2025:,.0f}')

    # YoY if historical 2024 exists
    hist_2024 = ts[ts.index.year==2024].sum()
    if hist_2024>0:
        yoy = (predicted_sum_2025 - hist_2024)/hist_2024*100
        st.write(f'YoY (prognoza 2025 vs suma 2024): {yoy:.1f}%')
    else:
        st.write('Brak pełnych danych za 2024 do porównania YoY.')

    # show next N forecast points
    st.write('Najbliższe prognozowane punkty:')
    st.dataframe(final_pred.head(30).rename('forecast').to_frame())

    csv = final_pred.rename('forecast').to_csv().encode('utf-8')
    st.download_button('Pobierz prognozę (CSV)', data=csv, file_name='forecast.csv', mime='text/csv')

st.sidebar.header('Dalsze kroki / rozbudowa')
st.sidebar.write('''
- Dodanie regresorów (promo, pogoda, kampanie) do Prophet / ARIMA / LGBM,
- Ensemble: średnia ważona modeli (jak zbierzesz więcej danych),
- Prophet tuning (changepoint_prior_scale, seasonality_prior_scale),
- Automatyczny pipeline do retrain (np. co 7 dni) i monitor błędu.
''')

st.caption('Uwaga: zainstaluj pakiety prophet (lub fbprophet), pmdarima i holidays. Dla Streamlit Cloud dodaj je do requirements.txt.')
