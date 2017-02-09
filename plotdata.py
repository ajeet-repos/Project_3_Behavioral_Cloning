import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the data and splitting them into trianing and validation set.
driving_data = pd.read_csv('../driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])
steering_data = driving_data['steering'].values

X_train, X_val, y_train, y_val = train_test_split(driving_data, steering_data, test_size=0.2, random_state=42)


n, bins, patches = plt.hist(y_train, alpha=0.75)
plt.plot(bins, linewidth=1)
plt.axis([-2, 2, 0, 15000])
plt.grid(True)
plt.show()
