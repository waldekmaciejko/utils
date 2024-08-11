import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

x = np.linspace(-10, 10) 
y = x**2 + np.random.uniform(size=50, low=-10, high=10)
yy = x**2
yyy = x + 2

matplotlib.rcParams.update({'font.size': 22})
fig, ax = plt.subplots()
plt.rcParams.update({'font.size': 22})
ax.plot(x, y, 'o', label='data')
ax.plot(x, yy, 'r', label='good learnt')
ax.plot(x,y, label='high variance')
ax.plot(x, yyy, label='high bias')
plt.legend()
plt.show()