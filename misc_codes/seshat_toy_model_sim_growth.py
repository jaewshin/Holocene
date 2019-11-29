#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:42:18 2018
Checked on May 16 2019

The code to run a toy model for figure 10 and 11 (simulated CC growth) on Arxiv version.

@author: hajime
"""



import numpy as np
from matplotlib import pyplot as plt, cm as cm, mlab as mlab

#x = np.linspace(-4,4,num=2000)
#x = np.random.normal(2.0,2.0,2000)
#x = 5-np.random.exponential(.1,2000)
x = -np.random.chisquare(2,4000)
#x = np.random.triangular(0.,4.,4.,2000)
x_lin = np.linspace(x.min(),x.max(),num=4000)

n, bins, patches = plt.hist(x, 100,  facecolor='blue', alpha=0.5)
plt.title('x')
plt.show()
plt.close()



#S1:
'''
L1=1.
k1=1.
x_start1=-10.
L2=5
k2=1.
x_start2=-1.#S1
'''

L1=1.
k1=1.
#x_start1=-10.#S1
x_start1=-10.
y1 = L1/(1.+np.exp(-k1*(x-x_start1 )))
y1_lin = L1/(1.+np.exp(-k1*(x_lin-x_start1 )))

plt.plot(x_lin,y1_lin)
plt.title('First CC growth')
plt.show()
plt.close()

n, bins, patches = plt.hist(y1, 200,  facecolor='blue', alpha=0.5)
plt.title('CC1')
plt.show()
plt.close()



L2=5.
k2=1.
x_start2=-5.#S1
#x_start2=9.0
y2 = L2/(1.+np.exp(-k2*(x-x_start2 )))
y2_lin = L2/(1.+np.exp(-k2*(x_lin-x_start2 )))

#k_exp = .3
#y2 = np.exp(k_exp*x)
#y2_lin = np.exp(k_exp*x_lin)

plt.plot(x_lin,y2_lin)
plt.title('Second CC growth')
plt.show()
plt.close()

n, bins, patches = plt.hist(y2, 200,  facecolor='blue', alpha=0.5)
plt.title('CC2')
plt.show()
plt.close()

Y=.5*y1+.5*y2
Y_lin=.5*y1_lin+.5*y2_lin
plt.plot(x_lin,Y_lin)
plt.title('Growth of the average')
plt.show()
plt.close()

n, bins, patches = plt.hist(Y, 200,  facecolor='blue', alpha=0.5)
plt.title('Histogram of the average of two CC')
plt.show()
plt.close()
