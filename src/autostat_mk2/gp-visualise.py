import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

data = []

src = sys.argv[1]
title = r"GP Regression. Kernel: " + str(sys.argv[2]) + r". Hyperparameters: " + str(sys.argv[3])
with open(src) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(row)
            
x = list(map(float, data[0]))
y = list(map(float, data[1]))
x_star = list(map(float, data[2]))
f_star = list(map(float, data[3]))
var = list(map(float, data[4]))

sd2 = 2*np.sqrt(var)

fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(x, y, 'ko',lw=2.0)
plt.plot(x_star, f_star, 'black', lw=2.0)
plt.fill_between(x_star, f_star-sd2, f_star+sd2, color='grey', alpha=0.5)
plt.grid()
plt.ylabel(r"$y$", fontsize=16)
plt.xlabel(r"$x_1$", fontsize=16)
fig.canvas.set_window_title("GP Plot")
plt.title(title, fontsize=16)
plt.show()
