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