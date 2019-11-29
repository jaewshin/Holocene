import time
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal, norm
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
from collections import defaultdict
import os
import math
import time
import pickle
from sklearn.decomposition import PCA, FastICA
from sklearn import linear_model
import scipy.spatial as spatial
from IPython.core.debugger import Pdb
from seshat import *

worldRegions,NGAs,PC_matrix,CC_df, CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()

# The x-value for out data
x_out = movArrayOut[:,:,0]
dx_out = movArrayOut[:,:,1]
tau_out = velArrayOut[:,:,2]

np.savetxt('x_out.csv',x_out,delimiter=',')
np.savetxt('dx_out.csv',dx_out,delimiter=',')
np.savetxt('tau_out.csv',tau_out,delimiter=',')
