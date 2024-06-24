import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL
from scipy.fft import fft, fftfreq
import numpy as np



#complete series
idx = pd.date_range(min(maindatasetF['datum']), max(maindatasetF['datum']), freq='D')
maindatasetF = maindatasetF.sort_values(['datum'])
maindatasetF.index = pd.DatetimeIndex(maindatasetF['datum'])
maindatasetF = maindatasetF.reindex(idx, fill_value=0)
maindatasetF = maindatasetF[['N02BE']]
maindatasetF

#fourier transforms
from scipy.fft import fft, fftfreq
import numpy as np
yf = fft(ydf['GASREGW'].values)
N = len(ydf)
xf = 1/(fftfreq(N, d=1.0))
nyf = np.abs(yf)
four = pd.DataFrame({'Period':xf, 'Amp':nyf})
four = four[(four['Period']>0) & (four['Period']<=200)]
four = four.sort_values(['Period'], ascending=True)
four['Period'] = four['Period'].apply(lambda x: math.floor(x))
four = four.groupby('Period').max().reset_index(drop=False)
four = four.sort_values('Amp', ascending=False).head(5)
four


#with statsmodels
from statsmodels.tsa.seasonal import MSTL
seas = MSTL(maindatasetF, periods=(7, 7*4, 180, 365)).fit()
seas.plot()
plt.tight_layout()
plt.show()


#with prophet

#fill data
df_ = df_.fillna(0)
df_ = df_.groupby(by=['id','wo_createddate'], as_index=False)['wo_workordernumber'].nunique()
firstdate = min(df_["wo_createddate"]) + pd.Timedelta(days=30) - MonthBegin(1)
lastdate = max(df_["wo_createddate"]) - pd.Timedelta(days=30) + MonthEnd(1)
firstdate = firstdate.strftime("%Y-%m-%d")
lastdate = lastdate.strftime("%Y-%m-%d")
df_ = df_[df_['wo_createddate'] >pd.to_datetime(firstdate).date()]
df_ = df_[df_['wo_createddate'] <pd.to_datetime(lastdate).date()]
df_['ds'] = pd.to_datetime(df_['wo_createddate'])
df_['y'] = (df_['wo_workordernumber']+0).values 
df_ = df_[['id','ds','y']]

#for an id
df_short = df_[df_['id']==int_id]
df_short = df_short[['ds','y']]
idx = pd.date_range(firstdate, lastdate)
df_short = df_short.set_index('ds').reindex(idx, fill_value=0).reset_index(drop=False)
df_short = df_short.rename(columns={'index': 'ds'}) 
df_short = df_short.fillna(0)
df_short = df_short.resample(rule='W', on='ds')['y'].sum().reset_index(drop=False)
cap = max(df_short['y'])
df_short["cap"] = cap
floor = min(df_short['y'])
df_short["floor"] = floor
y = df_short["y"].to_numpy()

#fourier for seasonality
yf = fft(y).real
xf = np.arange(0,y.size)
fourier = pd.DataFrame([xf,yf]).T
fourier = fourier.sort_values(by=1, ascending=False)
fourier = fourier[fourier[1]>0]
fourier = fourier[0:20]

#define prophet and add seasonality (can also add holidays and etc..)
m = Prophet(changepoint_prior_scale=0.15,changepoint_range=0.9)
m.add_seasonality(name='monthly', period=5, fourier_order=5)
if len(fourier[(fourier[0]>8) & (fourier[0]<12)])>0:
        m.add_seasonality(name='bimonthly', period=10, fourier_order=5)
if len(fourier[(fourier[0]>23) & (fourier[0]<29)])>0:
        m.add_seasonality(name='semesterly', period=26, fourier_order=5)
if len(fourier[(fourier[0]>48) & (fourier[0]<55)])>0:
        m.add_seasonality(name='year', period=52, fourier_order=5)
if len(fourier[(fourier[0]>100) & (fourier[0]<108)])>0:
        m.add_seasonality(name='biyear', period=104, fourier_order=5)
if len(fourier[(fourier[0]>11) & (fourier[0]<15)])>0:
        m.add_seasonality(name='quarterly', period=13, fourier_order=5)

#fit and forecast        
m.fit(df_short)
future = m.make_future_dataframe(periods=28, freq='W')
forecast = m.predict(future)
result = pd.merge(forecast, df_short, how="left", on=["ds"])
result = result.fillna(0)

#format results (are messy)
result['integration_id'] = int_id
result['year'] = result.set_index('ds').index.year
result['month'] = result.set_index('ds').index.month
result = result.fillna(0)  
monthly = result.groupby(by=['year','month']).mean().reset_index(drop=False)
monthly = monthly[['integration_id','year','month','yhat','y']].rename(columns={'yhat': 'yhat_month', 'y': 'y_month'}) 
current = monthly[(monthly['year']==curr_year) & (monthly['month']==curr_month)]
current = current[['integration_id','yhat_month','y_month']].rename(columns={'yhat_month': 'yhat_current', 'y_month': 'y_current'}) 
result = pd.merge(result, current, on=["integration_id"])
seasons = ['monthly','bimonthly','semesterly','year','biyear','quarterly']
valid_cols = [c for c in seasons if c in result]
result['seasonalities'] = result[valid_cols].mean(axis=1)
result = result[['integration_id','ds','trend','yhat','y','yhat_current','y_current','seasonalities']]
result['Prediction'] = np.where(result['yhat']>0, result['yhat'].round(), 0)

