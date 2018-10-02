# HW1
Data science homework 1
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Read the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.names"
"""
Attribute Information:
     1. L-CORE (patient's internal temperature in C):
              high (> 37), mid (>= 36 and <= 37), low (< 36)
     2. L-SURF (patient's surface temperature in C):
              high (> 36.5), mid (>= 36.5 and <= 35), low (< 35)
     3. L-O2 (oxygen saturation in %):
              excellent (>= 98), good (>= 90 and < 98),
              fair (>= 80 and < 90), poor (< 80)
     4. L-BP (last measurement of blood pressure):
              high (> 130/90), mid (<= 130/90 and >= 90/70), low (< 90/70)
     5. SURF-STBL (stability of patient's surface temperature):
              stable, mod-stable, unstable
     6. CORE-STBL (stability of patient's core temperature)
              stable, mod-stable, unstable
     7. BP-STBL (stability of patient's blood pressure)
              stable, mod-stable, unstable
     8. COMFORT (patient's perceived comfort at discharge, measured as
              an integer between 0 and 20)
     9. decision ADM-DECS (discharge decision):
              I (patient sent to Intensive Care Unit),
              S (patient prepared to go home),
              A (patient sent to general hospital floor)"""

names = ['L-core','L-surf', 'L-O2','L-BP','Surf-STBL','Core-STBL', 'BP-STBL', 'Confort', 'Decision']
df = c = pd.read_cvs(url, names = names, comment='#')

plt.figure(); df.plot();

from io import StringIO
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='0', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data
