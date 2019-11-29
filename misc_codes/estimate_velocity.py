"""
Attempts to estimate flow field.

@author: hajime
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, PolynomialFeatures
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal, norm

from scipy.interpolate import SmoothBivariateSpline
from scipy.spatial import ConvexHull
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

import time
import pickle
import os
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.patches import Patch

import tensorflow as tf
import tensorflow_probability as tfp
from seshat import *

from multiprocessing import Pool 
import multiprocessing as mp

from scipy.io import savemat
from scipy.special import kl_div
from scipy.optimize import fsolve,root,minimize,fmin
#worldRegions,NGAs,PC_matrix,CC_df,PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()

from seshat_NN import Est_Flow_NN_CV,Est_Flow_NN

#Read data generated in seshat.py
worldRegions,NGAs,PC_matrix,CC_df,CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()



#tfd = tfp.distributions
sigma2_list = np.tile(np.geomspace(.25,2.5,num=10,endpoint=True).reshape([-1,1]),[1,9])
#sigma2 = np.power( ( (4/(dim+2))**(1/(dim+4)) )*(n**(-1/(dim+4))) *sig_xy, 2) Silverman's rule
u0Vect_out,v0Vect_out = createGridForPC12(dGrid,velArrayOut)
x_grids_out =  np.concatenate( (np.repeat(u0Vect_out,len(v0Vect_out)).reshape([-1,1]), np.tile( v0Vect_out, len(u0Vect_out) ).reshape([-1,1]) ) ,axis=1)
u0Vect_in,v0Vect_in = createGridForPC12(dGrid,velArrayIn)
x_grids_in =  np.concatenate( (np.repeat(u0Vect_in,len(v0Vect_in)).reshape([-1,1]), np.tile( v0Vect_in, len(u0Vect_in) ).reshape([-1,1]) ) ,axis=1)


#%%
flowInfo['NGA_id']=flowInfo.groupby('NGA').ngroup()
flowInfo['ID_within_NGA'] = flowInfo.groupby('NGA_id')['NGA_id'].rank(method='first')

#Extract the necessary data
##"OUT" data
d_x_out = velArrayOut[:,:2,0] #The starting point of "OUT" vector in the first 2 PC space
d_y_out = velArrayOut[:,2:,0] #The starting point of "OUT" vector in the other 7 PC space
d_v_out = velArrayOut[:,:2,1] #The "OUT" velocity in the first 2 PC space
d_w_out = velArrayOut[:,2:,1] #The "OUT" velocity in the other 7 PC space
d_xy_out = velArrayOut[:,:,0] #The starting point of "OUT" vector in 9 PC space

pos_v_not_nan_out = np.where(~np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of non-NaN points due to end point
pos_v_nan_out = np.where(np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of NaN points due to end point

n_obs_out = len(pos_v_not_nan_out)

#d_xy_tf_out = tf.constant(d_xy_out[pos_v_not_nan_out,:],dtype=tf.float32)
#d_v_tf_out = tf.constant(d_v_out[pos_v_not_nan_out,:],dtype=tf.float32) #Removed NaN already

#Removing NaN
d_x_notnan_out = d_x_out[pos_v_not_nan_out,:]
d_xy_notnan_out = d_xy_out[pos_v_not_nan_out,:]
d_v_notnan_out = d_v_out[pos_v_not_nan_out,:]

#-----------------------------------------
#fit GMM
gmm_y_fit_out = GMM(n_components=2).fit(d_y_out)
cov_out = gmm_y_fit_out.covariances_
mean_out = gmm_y_fit_out.means_
weights_out = gmm_y_fit_out.weights_

#sample
'''
gmm_y_sample_out = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=weights_out),
    components_distribution=tfd.MultivariateNormalFullCovariance(
      loc=mean_out,       # One for each component.
      covariance_matrix=cov_out))  # And same here.
'''
sig_xy_out = np.std(d_xy_out,axis=0)
dim = 9
n_out = d_y_out.shape[0]




#%%
##"IN" data
d_x_in = velArrayIn[:,:2,0] #The ending point of "IN" vector in the first 2 PC space
d_y_in = velArrayIn[:,2:,0] #The ending point of "OUT" vector in the other 7 PC space
d_v_in = velArrayIn[:,:2,1] #The "IN" velocity in the first 2 PC space
d_w_in = velArrayIn[:,2:,1] #The "IN" velocity in the other 7 PC space
d_xy_in = velArrayIn[:,:,0] #The ending point of "OUT" vector in 9 PC space

pos_v_not_nan_in = np.where(~np.isnan(d_v_in))[0][::2].astype(np.int32) #Position of non-NaN points due to starting point
pos_v_nan_in = np.where(np.isnan(d_v_in))[0][::2].astype(np.int32) #Position of NaN points due to starting point

n_obs_in = len(pos_v_not_nan_in)

d_xy_tf_in = tf.constant(d_xy_in[pos_v_not_nan_in,:],dtype=tf.float32)
d_v_tf_in = tf.constant(d_v_in[pos_v_not_nan_in,:],dtype=tf.float32) #Removed NaN already

d_x_notnan_in = d_x_in[pos_v_not_nan_in,:]
d_xy_notnan_in = d_xy_in[pos_v_not_nan_in,:]
d_v_notnan_in = d_v_in[pos_v_not_nan_in,:]


#-----------------------------------------
'''
#fit GMM
gmm_y_fit_in = GMM(n_components=2).fit(d_y_in)
cov_in = gmm_y_fit_in.covariances_
mean_in = gmm_y_fit_in.means_s
weights_in = gmm_y_fit_in.weights_

#sample

gmm_y_sample_in = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=weights_in),
    components_distribution=tfd.MultivariateNormalFullCovariance(
      loc=mean_in,       # One for each component.
      covariance_matrix=cov_in))  # And same here.

sig_xy_in = np.std(d_xy_in,axis=0)
dim = 9
n_in = d_y_in.shape[0]
'''


#%%
#"Interpolated" data
d_x_interp = movArrayOutInterp[:,:2,0]
d_y_interp = movArrayOutInterp[:,2:,0]
d_v_interp = movArrayOutInterp[:,:2,1]
d_w_interp = movArrayOutInterp[:,2:,1]
d_xy_interp = movArrayOutInterp[:,:,0]

pos_v_not_nan_interp = np.where(~np.isnan(d_v_interp))[0][::2].astype(np.int32) #Due to end point

d_x_notnan_interp = d_x_interp[pos_v_not_nan_interp,:]
d_xy_notnan_interp = d_xy_interp[pos_v_not_nan_interp,:]
d_v_notnan_interp = d_v_interp[pos_v_not_nan_interp,:]











#%%
def normalize(data):
    mean = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    data_normalized = (data-mean)/std
    return data_normalized,mean,std



#d_x_pos:2D, d_x_dt:3D
d_x_pos = velArrayOut[:,:2,0 ]
dt = velArrayOut[:,0,2 ].reshape([-1,1])
d_x_dt = np.concatenate((d_x_pos,dt),axis=1)
d_mov = movArrayOut[:,:2,1]
d_v = velArrayOut[:,:2,1]

d_x_pos_notnan = d_x_pos[pos_v_not_nan_out]
d_x_dt_notnan = d_x_dt[pos_v_not_nan_out]
d_v_notnan = d_v[pos_v_not_nan_out]
d_mov_notnan = d_mov[pos_v_not_nan_out]

d_x_pos_normalized,d_x_pos_mean,d_x_pos_std = normalize(d_x_pos_notnan)

d_x_dt_normalized,d_x_dt_mean,d_x_dt_std = normalize(d_x_dt_notnan)

d_v_normalized,d_v_mean,d_v_std = normalize(d_v_notnan)

d_mov_normalized,d_mov_mean,d_mov_std = normalize(d_mov_notnan)

d_size = d_x_dt_notnan.shape[0]

x_grid_dt = np.concatenate( (x_grids_out, 100*np.ones([x_grids_out.shape[0],1]) ),axis=1 )

x_grid_pos_normalized = (x_grids_out-d_x_pos_mean)/d_x_pos_std
x_grid_dt_normalized = (x_grid_dt-d_x_dt_mean)/d_x_dt_std



#%%
#Graph Original Flow, with 500 year scale

'''
plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-8,7)
plt.ylim(-3,3)
scale=500.
for i in range(0,velArrayOut.shape[0]):
    if not np.isnan(velArrayOut[i,0,1]):
#        nn = [ind for ind,nga in enumerate(NGAs) if nga==ngaFlow[i]][0] 
#        plt.arrow(u[i],v[i],du[i],dv[i],width=.01,color=colors[nn])
        #rgb = cm.inferno(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
        rgb = cm.hsv(np.where(np.unique(flowInfo.NGA) == flowInfo.NGA[i])[0][0]/len(NGAs))
        #rgb = cm.gist_ncar(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
        #rgb = cm.jet(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
        #rgb = cm.prism(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
        #rgb = cm.brg(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
        plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,velArrayOut[i,1,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)

plt.axvline(x=-5.6762,color='k')
plt.axvline(x=3.9460,color='k')

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("OUT_in_the_data_500y.pdf")
plt.show()
plt.close()
'''
#%%
#Graph Original Flow
'''
if True:

    
    plt.figure(figsize=(17,8.85))
    plt.axis('scaled')
    plt.xlim(-6,5)
    plt.ylim(-3,3)
    scale=100.
    for i in range(0,velArrayOut.shape[0]):
        if not np.isnan(velArrayOut[i,0,1]):
    #        nn = [ind for ind,nga in enumerate(NGAs) if nga==ngaFlow[i]][0] 
    #        plt.arrow(u[i],v[i],du[i],dv[i],width=.01,color=colors[nn])
            #rgb = cm.inferno(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
            rgb = cm.hsv(np.where(np.unique(flowInfo.NGA) == flowInfo.NGA[i])[0][0]/len(NGAs))
            #rgb = cm.gist_ncar(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
            #rgb = cm.jet(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
            #rgb = cm.prism(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
            #rgb = cm.brg(np.where(np.unique(flowInfoInterp.NGA) == flowInfoInterp.NGA[i])[0][0]/len(NGAs))
            plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,velArrayOut[i,1,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    plt.savefig("OUT_in_the_data.pdf")
    #plt.show()
    plt.close()
    
    
'''




#%%
##Average value and velocity of PC2 in sliding window
'''
PC1=velArrayOut[:,0,0]
PC2=velArrayOut[:,1,0]
PC1_vel = velArrayOut[:,0,1]*100
PC2_vel = velArrayOut[:,1,1]*100

window_width =1.0
overlap = .5


score_list = []
vel_list = []
score_std_list = []
vel_std_list = []

score_error_list = []
vel_error_list = []


center_list = []

PC1_min = np.min(PC1)
PC1_max = np.max(PC1)

n_window = np.ceil( (PC1_max - PC1_min - window_width)/(window_width-overlap) ).astype(int)

for i in range(n_window):
    window = np.array([PC1_min+i*(window_width-overlap), PC1_min+i*(window_width-overlap)+window_width])
    center = np.mean(window)
    loc = (window[0]<=PC1) * (PC1<window[1])
    
    PC2_in_window = PC2[loc]
    PC2_vel_in_window = PC2_vel[loc]
    PC2_vel_in_window = PC2_vel_in_window[~np.isnan(PC2_vel_in_window)]
    
    score = np.mean(PC2_in_window)
    vel = np.mean(PC2_vel_in_window)
    score_std = np.std(PC2_in_window)
    vel_std = np.std(PC2_vel_in_window)

    score_error = score_std/np.sqrt(len(PC2_in_window) )
    vel_error = vel_std/np.sqrt( len(PC2_vel_in_window) )

    
    center_list.append(center)
    score_list.append(score)
    vel_list.append(vel)
    score_std_list.append(score_std)
    vel_std_list.append(vel_std)
    
    score_error_list.append(score_error)
    vel_error_list.append(vel_error)
   

plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, score_list,'b-o')
plt.errorbar(center_list, score_list, yerr=score_std_list)
plt.xlabel("PC1 (center of window)")
plt.ylabel("Average PC2")
plt.title("Average PC2 value(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC2 value(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()
    


plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, vel_list,'g-o')
plt.errorbar(center_list, vel_list, yerr=vel_std_list,color='green')
plt.xlabel("PC1 (center of window)")
plt.ylabel("Average velocity on PC2")
plt.title("Average velocity PC2 value(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average velocity PC2 value(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()




plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, score_list,'b-o')
plt.errorbar(center_list, score_list, yerr=score_error_list)
plt.xlabel("PC1 (center of window)")
plt.ylabel("Average PC2")
plt.title("Average PC2 value with error(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC2 value with error bar(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()
    


plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, vel_list,'g-o')
plt.errorbar(center_list, vel_list, yerr=vel_error_list,color='green')
plt.xlabel("PC1 (center of window)")
plt.ylabel("Average velocity on PC2")
plt.title("Average velocity PC2 value with error(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average velocity PC2 value with error(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()

'''
#%%
'''
PC2 on x-axis
'''
'''
score_list = []
vel_list = []
score_std_list = []
vel_std_list = []

score_error_list = []
vel_error_list = []


center_list = []

PC2_min = np.min(PC2)
PC2_max = np.max(PC2)

n_window = np.ceil( (PC2_max - PC2_min - window_width)/(window_width-overlap) ).astype(int)

for i in range(n_window):
    window = np.array([PC2_min+i*(window_width-overlap), PC2_min+i*(window_width-overlap)+window_width])
    center = np.mean(window)
    loc = (window[0]<=PC2) * (PC2<window[1])
    
    PC1_in_window = PC1[loc]
    PC1_vel_in_window = PC1_vel[loc]
    PC1_vel_in_window = PC1_vel_in_window[~np.isnan(PC1_vel_in_window)]
    
    score = np.mean(PC1_in_window)
    vel = np.mean(PC1_vel_in_window)
    score_std = np.std(PC1_in_window)
    vel_std = np.std(PC1_vel_in_window)

    score_error = score_std/np.sqrt(len(PC2_in_window) )
    vel_error = vel_std/np.sqrt( len(PC2_vel_in_window) )
    
    center_list.append(center)
    score_list.append(score)
    vel_list.append(vel)
    score_std_list.append(score_std)
    vel_std_list.append(vel_std)

    score_error_list.append(score_error)
    vel_error_list.append(vel_error)

plt.axis()
plt.xlim(-2.3,2.3)
plt.ylim(-2.7,2.)
plt.plot(center_list, score_list,'b-o')
plt.errorbar(center_list, score_list, yerr=score_std_list)
plt.xlabel("PC2 (center of window)")
plt.ylabel("Average PC1")
plt.title("Average PC1 value(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC1 value(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()
    


plt.axis()
plt.xlim(-2.3,2.3)
plt.ylim(-1.,1.5)
plt.plot(center_list, vel_list,'g-o')
plt.errorbar(center_list, vel_list, vel_std_list,color='green')
plt.xlabel("PC2 (center of window)")
plt.ylabel("Average velocity on PC1")
plt.title("Average velocity PC1 value(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average velocity PC1 value(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()



plt.axis()
plt.xlim(-2.3,2.3)
plt.ylim(-2.7,2.)
plt.plot(center_list, score_list,'b-o')
plt.errorbar(center_list, score_list, yerr=score_error_list)
plt.xlabel("PC2 (center of window)")
plt.ylabel("Average PC1")
plt.title("Average PC1 value with error(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC1 value with error(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()
    


plt.axis()
plt.xlim(-2.3,2.3)
plt.ylim(-1.,1.5)
plt.plot(center_list, vel_list,'g-o')
plt.errorbar(center_list, vel_list, vel_error_list,color='green')
plt.xlabel("PC2 (center of window)")
plt.ylabel("Average velocity on PC1")
plt.title("Average velocity PC1 value with error(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average velocity PC1 value with error(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()
'''

#%%
'''
import copy


CC_scaled_df = pd.DataFrame(CC_scaled,columns=[ 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money' ])
CC_scaled_df[['NGA','Time']] = CC_df[['NGA','Time']]


CC_reshape = CC_scaled_df.groupby(['NGA','Time']).mean().reset_index()
CC_fwd = CC_reshape.groupby(['NGA']).shift(-1)

CC_out = CC_fwd[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']] - CC_reshape[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']] 
CC_out['NGA'] = CC_reshape['NGA']

CC_out_vel = CC_out[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']].div(CC_out['Time'],axis=0) *100.
CC_out_vel.columns = [str(col) + '_vel' for col in CC_out_vel.columns]
CC_out_vel['NGA'] = CC_out['NGA']



cc_all = CC_reshape.drop(columns=['NGA','Time']).values
pc_all = velArrayOut[:,:,0]
PC1 = pc_all[:,0]
PC2 = pc_all[:,1]


lr_pc1 = LinearRegression(fit_intercept=False)
lr_pc1.fit(cc_all, pc_all[:,0].reshape([-1,1]))
pc1_coeff = lr_pc1.coef_

lr_pc2 = LinearRegression(fit_intercept=False)
lr_pc2.fit(cc_all, pc_all[:,1].reshape([-1,1]))
pc2_coeff = lr_pc2.coef_

#negative coefficient CC:  'PolPop', 'PolTerr', 'CapPop', 'levels'
#positive coefficient CC:  ('government','infrastr', 'writing', 'texts', 'money')
pc2_coeff_negative = copy.deepcopy(pc2_coeff)
pc2_coeff_negative[0,4:]=0. #zero-out the positive coefficient CCs. 
pc2_coeff_positive = copy.deepcopy(pc2_coeff)
pc2_coeff_positive[0,:4]=0. #zero-out the negative coefficient CCs ('PolPop', 'PolTerr', 'CapPop', 'levels')

pc2_negativeCC_only = np.dot( cc_all, pc2_coeff_negative.T ).flatten()
pc2_positiveCC_only = np.dot( cc_all, pc2_coeff_positive.T ).flatten()


plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,4)
scale=100.
for i in range(0,velArrayOut.shape[0]):
    if not np.isnan(velArrayOut[i,0,1]):
        rgb = cm.hsv(np.where(np.unique(flowInfo.NGA) == flowInfo.NGA[i])[0][0]/len(NGAs))
        plt.arrow(velArrayOut[i,0,0],pc2_negativeCC_only[i],velArrayOut[i,0,1]*scale,(pc2_negativeCC_only[i+1]-pc2_negativeCC_only[i]),width=.01,head_width=.06,head_length=.04,color=rgb)
plt.xlabel("PC1")
plt.ylabel("PC2 with negative-loading CC")

plt.savefig("OUT_in_the_data_PC2_NegativeLoadingCC.pdf")
#plt.show()
plt.close()




plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,4)
scale=100.
for i in range(0,velArrayOut.shape[0]):
    if not np.isnan(velArrayOut[i,0,1]):
        rgb = cm.hsv(np.where(np.unique(flowInfo.NGA) == flowInfo.NGA[i])[0][0]/len(NGAs))
        plt.arrow(velArrayOut[i,0,0],pc2_positiveCC_only[i],velArrayOut[i,0,1]*scale,(pc2_positiveCC_only[i+1]-pc2_positiveCC_only[i]),width=.01,head_width=.06,head_length=.04,color=rgb)
plt.xlabel("PC1")
plt.ylabel("PC2 with positive-loading CC")

plt.savefig("OUT_in_the_data_PC2_PositiveLoadingCC.pdf")
#plt.show()
plt.close()
'''
#%%

'''
#Each NGA
scale=100.
for nga in np.unique(flowInfo.NGA):
    plt.figure(figsize=(17,8.85))
    plt.axis('scaled')
    plt.xlim(-6,5)
    plt.ylim(-3,4)
    for i in range(0,velArrayOut.shape[0]):
        if not np.isnan(velArrayOut[i,0,1]):
            if flowInfo.NGA[i]==nga:
                rgb = cm.hsv(np.where(np.unique(flowInfo.NGA) == flowInfo.NGA[i])[0][0]/len(NGAs))
                plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,(pc2_negativeCC_only[i+1]-pc2_negativeCC_only[i]),width=.002,head_width=.03,head_length=.02, linestyle='dashed')
                plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,(pc2_positiveCC_only[i+1]-pc2_positiveCC_only[i]),width=.002,head_width=.03,head_length=.02, linestyle='dashed')
                plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,velArrayOut[i,1,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)
    plt.xlabel("PC1")
    plt.ylabel("PC2 with positive-loading CC")
    #plt.savefig("OUT_in_the_data_PC2_PositiveLoadingCC.pdf")
    plt.show()
    plt.close()
'''

#%%
'''
window_width=1.
overlap=.5
pc2_negative_mean_list = []
pc2_positive_mean_list = []
pc2_mean_list = []
center_list = []

PC1_min = np.min(PC1)
PC1_max = np.max(PC1)

n_window = np.ceil( (PC1_max - PC1_min - window_width)/(window_width-overlap) ).astype(int)

for i in range(n_window):
    window = np.array([PC1_min+i*(window_width-overlap), PC1_min+i*(window_width-overlap)+window_width])
    center = np.mean(window)
    loc = (window[0]<=PC1) * (PC1<window[1])
    
    pc2_negative_mean_list.append(np.mean(pc2_negativeCC_only[loc])  )
    pc2_positive_mean_list.append(np.mean(pc2_positiveCC_only[loc])  )
    pc2_mean_list.append( np.mean( PC2[loc] ) )    
    
    center_list.append(center)
    
    

plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, pc2_negative_mean_list)
plt.xlabel("PC1 (center of window)")
plt.ylabel("average PC2 with subset of CC")
plt.title("Average PC2 with negative loading CC(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC2 with negative loading CC(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()


plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, pc2_positive_mean_list)
plt.xlabel("PC1 (center of window)")
plt.ylabel("average PC2 with subset of CC")
plt.title("Average PC2 with positive loading CC(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC2 with positive loading CC(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))#
#plt.show()
plt.close()



plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, pc2_negative_mean_list, label='negative CC')
plt.plot(center_list, pc2_positive_mean_list, label='positive CC')
plt.plot(center_list, pc2_mean_list, label='Overall')

plt.xlabel("PC1 (center of window)")
plt.ylabel("average PC2 with subset of CC")
plt.legend()
plt.title("Average PC2 with subset of CC(window %.3f, step %.3f)"%(window_width,window_width-overlap))
plt.savefig("Average PC2 with subset of CC(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
#plt.show()
plt.close()

'''




#%%
## NN with time gap
if True:
    param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[50000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[.0],'l2_w':[.0,.001,.01,.1,1.,10.]}]
    param_list = list(ParameterGrid(param_for_grid))
    
    for i in range(len(param_list)):
        if param_list[i]['layer'] is 1:
            param_list[i]['wid2'] = 0
    param_list = list(map(dict, set(tuple(sorted(d.items())) for d in param_list)) )
    
    error = np.zeros(len(param_list))   
    pred_grid = [] 
    
    for i in range(len(param_list)):
        print("---Setting"+str(i)+' of '+str(len(param_list))+'---')
        print(param_list[i])
        dim = param_list[i]['dim']
    
        print('---CV----')
        print("---Setting"+str(i)+'---')
        print(param_list[i])
        error[i] = Est_Flow_NN_CV(d_x=d_x_dt_normalized,d_v=d_mov_normalized,param=param_list[i])    
        _, pred = Est_Flow_NN(d_x=d_x_dt_normalized,d_v=d_mov_normalized,x_to_pred=x_grid_dt_normalized ,**param_list[i]) 
            
        pred = pred*d_mov_std+d_mov_mean #De-normalized the predicted flow
        
        pred_grid.append(pred)
    
    
    with open("CV_result_NN_dt_190422_L2.pickle","wb") as f:
        pickle.dump(error, f)
    with open("CV_param_NN_dt_190422_L2.pickle","wb") as f:
        pickle.dump(param_list, f)
    with open("CV_pred_NN_dt_190422_L2.pickle","wb") as f:
        pickle.dump(pred_grid, f)
    

#%%
param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[50000],'LEARNING_RATE':[1e-3],'dim':[2],'lag':[1],'l1_w':[.0],'l2_w':[.0,.001,.01,.1,1.,10.]}]

param_list = list(ParameterGrid(param_for_grid))

for i in range(len(param_list)):
    if param_list[i]['layer'] is 1:
        param_list[i]['wid2'] = 0
param_list = list(map(dict, set(tuple(sorted(d.items())) for d in param_list)) )


if True:
    error = np.zeros(len(param_list))  
    pred_grid = [] 
    for i in range(len(param_list)):
        print("---Setting"+str(i)+' of '+str(len(param_list))+'---')
        print(param_list[i])
        dim = param_list[i]['dim']
        
    
        print('---CV----')
        print("---Setting"+str(i)+'---')
        print(param_list[i])
        
        start_time = time.time()
    
        
        error[i] = Est_Flow_NN_CV(d_x=d_x_pos_normalized,d_v=d_v_normalized,param=param_list[i])  
        _, pred = Est_Flow_NN(d_x=d_x_pos_normalized,d_v=d_v_normalized,x_to_pred=x_grid_pos_normalized ,**param_list[i])  
        pred_grid.append(pred)
        
        pred = pred*d_v_std+d_v_mean
    
    
        end_time = time.time()
        time_elapsed = end_time-start_time
        print('time for a setting: ',time_elapsed)
    
        
    with open("CV_result_NN_nodt_190422_L2.pickle","wb") as f:
        pickle.dump(error, f)
    with open("CV_pred_NN_nodt_190422_L2.pickle","wb") as f:
        pickle.dump(pred_grid, f)
    with open("CV_param_NN_nodt_190422_L2.pickle","wb") as f:
        pickle.dump(param_list, f)


