import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20,10
df3 = pd.read_csv('Apple.csv')
print(df3.head(10))
def hello():
	return df3.head(10)