import numpy as np
import pandas as pd
#import datetime, copy, imp
import pickle
import time
import os
import re
from sklearn.model_selection import StratifiedKFold
from importlib import reload


from tqdm.auto import tqdm, trange
from tqdm.notebook import tqdm
tqdm.pandas()

import sys
sys.path.insert(0, '../util/')

# Data description


dataFileStr = '../data/MLBHospitalData.hd5'
dat = pd.read_hdf(dataFileStr,key='Data')
print(dat)

# List of the diffirent states a patient can be in


print (dat.Event.unique())

# Population of the data


n = len(dat.index.get_level_values(0).unique())
print('Population: {} individuals'.format(n))

entryDates = dat.groupby(level=0).apply(lambda x: x.Date.min())
entryDateCount = pd.Series(range(n),index=entryDates).resample('1M').count()
entryDateCount.plot()

# summary of each individual's uique events


def timeline_summary(tbl,startDate='NoDate',endDate='NoDate'):
    if startDate != 'NoDate' and endDate != 'NoDate':
        tbl = tbl.loc[ (tbl.Date >= startDate) & (tbl.Date <= endDate) ]
        
    return pd.Series({
        'NumGoodTestResult': (tbl.Event == 'GoodTestResult').sum(),
        'NumStay': (tbl.Event == 'Stay').sum(),
        'NumBadTestResult': (tbl.Event == 'BadTestResult').sum(),
        'NumVitalsCrash': (tbl.Event == 'VitalsCrash').sum(),
        'Tenure': (tbl.Date.max()-tbl.Date.min()).days
    })

ftr = dat.groupby(level=0).progress_apply(timeline_summary)

print(ftr)