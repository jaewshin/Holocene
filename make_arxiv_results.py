import time
from collections import defaultdict
import os
import math
import time
import pickle
import copy 
import sys

import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal, norm
import scipy.spatial as spatial
import tensorflow as tf
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
from IPython.core.debugger import Pdb

from seshat import *
    
# Call generate_core_data (see seshat module defined in seshat.py) to load data for subsequent analyses
print('Calling generate_core_data')
worldRegions,NGAs,PC_matrix,CC_df, CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo, movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()

# # create a directory to store all the generated plots
createFolder('figures')

####################################################
# PCA on data drawn from sinuid function (Figure 1)
####################################################

def x(t):
	return t

def y(t):
	return np.sin(2*t)

def draw_vector(v0, v1, ax=None, color='r', label=''):
	ax = ax or plt.gca()
	arrowprops=dict(arrowstyle='->',
		linewidth=2,
		shrinkA=0, shrinkB=0, color=color)
	ax.annotate('', v1, v0, arrowprops=arrowprops, label=label)

t_range = np.linspace(-7, 7, 100)
xval = x(t_range)
yval = y(t_range)

xval = xval.reshape((-1, 1))
yval = yval.reshape((-1, 1))
data = np.hstack((xval, yval))

pca = PCA(n_components=2, svd_solver='full')
pca.fit(data)

fig, ax = plt.subplots()
plt.scatter(xval, yval)
plt.plot(xval, yval, linewidth=1)

i=0

for length, vector in zip(pca.explained_variance_, pca.components_):
	v = vector * 1 * np.sqrt(length)

	if i==0:
		color = 'r'
		label = 'PC1'
		draw_vector(pca.mean_, pca.mean_ + v, color=color, label=label)
		draw_vector(pca.mean_, pca.mean_ - v, color=color)

	else:
		color = 'g'
		label = 'PC2'
		a2 = draw_vector(pca.mean_, pca.mean_ + v, color=color, label=label)
		draw_vector(pca.mean_, pca.mean_ - v, color=color)

	i+=1

red_patch = mpatches.Patch(color='r', label='First Principal Component')
blue_patch = mpatches.Patch(color='g', label='Second Principal Component')

plt.legend(handles=[red_patch, blue_patch])
plt.xlim(-7,7)
plt.ylim(-3,3)
plt.savefig(os.path.join('figures', "sinuid_PCA.pdf"))
plt.close()
print("Done with generating a plot where PCA is not informative (Figure 1)")

###############################################################################
# Average score of observations on PC2 in a sliding window along PC1 (Figure 2)
###############################################################################

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
plt.plot(center_list, score_list, 'b-o')
plt.errorbar(center_list, score_list, yerr=score_error_list, capthick=2, capsize=3)
plt.xlabel("PC1 (center of window)")
plt.ylabel("Average PC2")
plt.savefig(os.path.join('figures', "Average_PC2_value.pdf"))
plt.close()
print("Done with average pc2 value plot (Figure 2)")

##########################################################################################
# Average PC value for PC2 with subset of CCs (Figure 3)
##########################################################################################

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

pc2_coeff_negative = copy.deepcopy(pc2_coeff)
pc2_coeff_negative[0,4:]=0.
pc2_coeff_positive = copy.deepcopy(pc2_coeff)
pc2_coeff_positive[0,:4]=0.

pc2_negativeCC_only = np.dot( cc_all, pc2_coeff_negative.T ).flatten()
pc2_positiveCC_only = np.dot( cc_all, pc2_coeff_positive.T ).flatten()

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
plt.plot(center_list, pc2_negative_mean_list, label='negative PC2 components')
plt.plot(center_list, pc2_positive_mean_list, label='positive PC2 components')
plt.plot(center_list, pc2_mean_list, label='all 9 components')

plt.xlabel("PC1 (center of window)")
plt.ylabel("sum over CCs with positive /negative PC2 components")
plt.legend()
# plt.title("Average PC2 with subset of CC(window %.3f, step %.3f)"%(window_width,window_width-overlap))
# plt.savefig("Average_PC2_with_subset_of_CC(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
plt.savefig(os.path.join('figures', "Average_PC2_with_subset_of_CC.pdf"))
plt.close()
print("Done with average pc2 subset plot (Figure 3)")

##############################################################################
# Creating movement plot [in the spirit of OUT_in_the_data] (Figure 4)
##############################################################################
plt.figure(figsize=(15,8))
plt.axis()
plt.xlim(-6,5)
plt.ylim(-3,3)
plt.xticks(size=25)
plt.yticks(size=25)

# plt.rc('xtick', labelsize=30)
# plt.rc('ytick', labelsize=30)
# plt.rc('font',    size=30)
pc3Min = min(movArrayOut[:,2,0][~np.isnan(movArrayOut[:,0,1])])
pc3Max = max(movArrayOut[:,2,0][~np.isnan(movArrayOut[:,0,1])])
newWorld = ['North America','South America']
# The following plotting parameters are used for all movement plots
lineWidth = .01
headWidth = .15
headLength = .1
arrowAlpha = 1 # Transparency
for i in range(0,movArrayOut.shape[0]):
    if not np.isnan(movArrayOut[i,0,1]):
        nga = flowInfo['NGA'][i]
        region = [key for key,value in regionDict.items() if nga in value][0]
        #rgb = cm.inferno((movArrayOut[i,2,0] - pc3Min) / (pc3Max - pc3Min))
        if region in newWorld:
          rgb = (1,0,0,arrowAlpha)
        else:
          rgb = (0,0,1,arrowAlpha)
        plt.arrow(movArrayOut[i,0,0],movArrayOut[i,1,0],movArrayOut[i,0,1],movArrayOut[i,1,1],
        		width=lineWidth,head_width=headWidth,head_length=headLength,color=rgb)
        # Next, plot interpolated points (if necessary)
        # Doing this all very explicitly to make the code clearer
        dt = velArrayOut[i,0,2]
        if dt > 100:
          for n in range(0,int(dt / 100) - 1):
            pc1 = movArrayOut[i,0,0] + velArrayOut[i,0,1]*(float(n+1))*100.
            pc2 = movArrayOut[i,1,0] + velArrayOut[i,1,1]*(float(n+1))*100.
            plt.scatter(pc1,pc2, s=5,  color=rgb)
#sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,norm=plt.Normalize(vmin=pc3Min, vmax=pc3Max))
#sm._A = []
#plt.colorbar(sm)

plt.xlabel("PC1", size=25)
plt.ylabel("PC2", size=25)
plt.savefig(os.path.join('figures', "pc12_movement_plot.pdf"))
plt.close()
print("Done with pc12 movmeent plot (Figure 4)")

###############################################################################
# Creating movement plot for Moralizing Gods (Figure 5)
###############################################################################
mgMissCol = 'grey'
mgAbsnCol = 'blue'
mgPresCol = 'green'
mgScatCol = 'red'

plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
plt.xticks(size=25)
plt.yticks(size=25)

mg_df = pd.read_csv('./mhg_code/data_used_for_nature_analysis.csv')
nat_df = pd.read_csv('./mhg_code/41586_2019_1043_MOESM6_ESM_sheet1.csv')
NGAs_nat = np.unique(nat_df['NGA'].values)
mg_tab = pd.DataFrame(columns=['NGA','Start','Stop','Value','Nature'])

mg_one_list=[]
for i in range(0,movArrayOut.shape[0]):
    if not np.isnan(movArrayOut[i,0,1]):
        nga = flowInfo['NGA'][i]
        time = flowInfo['Time'][i]
        if mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].shape[0]>0:
            if mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].values[0]==0:
                rgb = mgAbsnCol
            elif mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].values[0]==1:
                rgb = mgPresCol
                if not nga in mg_one_list:
                    rgb0 = 'orange'
                    if nga in NGAs_nat:
                        rgb0 = mgScatCol
                    else:
                        rgb0 = 'orange'
                    plt.scatter(velArrayOut[i,0,0],velArrayOut[i,1,0], color=rgb0,zorder=2)
                    mg_one_list.append( nga )
            else:
                rgb = mgMissCol

        plt.arrow(movArrayOut[i,0,0],movArrayOut[i,1,0],movArrayOut[i,0,1],movArrayOut[i,1,1],width=lineWidth,
        		head_width=headWidth,head_length=headLength,color=rgb,alpha=.5,zorder=1)
        
        # Next, plot interpolated points (if necessary)
        # Doing this all very explicitly to make the code clearer
        dt = velArrayOut[i,0,2]
        if dt > 100:
          for n in range(0,int(dt / 100) - 1):
            pc1 = movArrayOut[i,0,0] + velArrayOut[i,0,1]*(float(n+1))*100.
            pc2 = movArrayOut[i,1,0] + velArrayOut[i,1,1]*(float(n+1))*100.
            plt.scatter(pc1,pc2,s=10,color=rgb,alpha=.5,zorder=1)

plt.xlabel("PC1", size=25)
plt.ylabel("PC2", size=25)
plt.savefig(os.path.join('figures', "pc12_movement_plot_colored_by_MoralisingGods.pdf"))
plt.close()
print("Done with pc12 movement plot with mg (Figure 5)")


###############################################################################
# Generate histogram of PC1 for pooled imputations (Supplementary Figure 1)
###############################################################################
print('Making PC1 histogram')
num_bins = 50
n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, density=0, facecolor='blue', alpha=0.5)
#plt.title("Pooled over imputations")
plt.xlabel("Projection onto first Principal Component")
plt.ylabel("Counts")
#plt.legend()
fileStem = "pc1_histogram"
plt.savefig(os.path.join('figures', fileStem + ".pdf"))
plt.close()
print("Done with pc1 histogram plot (Figure 1")

###############################################################################
# Generate pc12 scatter plot (Supplementary Figure 2)
###############################################################################
print('Making PC1-PC2 scatter plot')
# Fit Gaussian Mixture
gmm = GMM(n_components=2).fit(PC_matrix)
cov = gmm.covariances_
prob_distr = gmm.predict_proba(PC_matrix)

# determine to which of the two gaussians each data point belongs by looking at probability distribution 
gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] >= prob_distr[i][1]]
gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] >= prob_distr[j][0]]

gauss1_time = [CC_times[i] for i in gauss1_idx] # time for the first gaussian data
gauss2_time = [CC_times[j] for j in gauss2_idx] # time for the second gaussian data

gauss1_pc1 = [PC_matrix[:,0][i] for i in gauss1_idx] # first pc values for the first gaussian
gauss2_pc1 = [PC_matrix[:,0][j] for j in gauss2_idx] # first pc values for the second gaussian

gauss1_pc2 = [PC_matrix[:,1][i] for i in gauss1_idx]
gauss2_pc2 = [PC_matrix[:,1][j] for j in gauss2_idx]

# Figure 2
plt.scatter(gauss1_pc1, gauss1_pc2, s=3, c='r')
plt.scatter(gauss2_pc1, gauss2_pc2, s=3, c='b')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(os.path.join('figures', 'pc12_scatter.pdf'), transparent=False)
plt.close()
print('Done with PC1-PC2 scatter plot (Supplementary Figure 2)')

##########################################################################################################
# plot for changing the threshold percentage of Gaussian probability distribution (Supplementary Figure 3)
##########################################################################################################

def flowvec_points(data, threshold):
    """
    Return the data points within each NGA where each point has less than 90% of belonging to either of the clusters
    """
    gmm = GMM(n_components=2).fit(data)
    cov = gmm.covariances_
    prob_distr = gmm.predict_proba(data)
    
    # determine to which of the two gaussians each data point belongs by looking at probability distribution 
    gauss_idx = [i for i in range(len(prob_distr)) 
                  if (prob_distr[i][0] <= threshold and prob_distr[i][1] <= threshold)]
    return gauss_idx

idx_len = []
range_val = []
threshold = '0.9'

for i in range(100):
    thres = float(threshold)
    idx = flowvec_points(CC_scaled, thres)
    idx_len.append(len(idx))
    range_val.append(thres)
    
    if len(idx) == 8280:
        break

    threshold += '9'
    
f, axes = plt.subplots(1, 2, sharey=True,figsize=(14,6))
axes[0].plot(range(len(range_val)), idx_len)
# axes[0].set_title('threshold_plot1', fontsize=10)
axes[0].set_xlabel('number of digits used')
# axes[1].set_ylabel('number of points included')

axes[0].set_ylabel('number of points included')
axes[1].plot([i*100 for i in range_val], idx_len)
# axes[1].set_title('threshold_plot2', fontsize=10)
axes[1].set_xlabel('percentage of threshold')
f.subplots_adjust(hspace=.5)

plt.savefig(os.path.join('figures', "threshold.pdf"))
plt.close()
print("Done with threshold plot (Supplementary Figure 3)")

#########################################################################################################
# Bootstrapping on the eigenvalues of each component and the angle between them (Supplementary Figure 4)
#########################################################################################################

def angle(vec1, vec2):
    """
    Given two vectors, compute the angle between the vectors
    """
    assert vec1.shape == vec2.shape
    
    cos_vec = np.inner(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    angle = math.acos(cos_vec)
    in_deg = math.degrees(angle)
    if in_deg >= 90:
        return (180-in_deg)
    return in_deg

def eig(mat):
    """
    Given a list of two matrices, compute the largest eigenvalues' proportion over sum of all eigenvalues as well 
    as the corresponding eigenvectors
    """
    eig_val1, eig_vec1 = np.linalg.eig(mat[0])
    eig_val2, eig_vec2 = np.linalg.eig(mat[1])
    
    assert eig_vec1.shape == eig_vec2.shape
    assert len(eig_val1) == len(eig_val2)
    
    # proportion of largest eigenvalue with respect to each component
    val1 = max(eig_val1)/sum(eig_val1)
    val2 = max(eig_val2)/sum(eig_val2)
    
    # eigenvector corresponding to the largest eigenvalue with respect to each component
    vec1 = eig_vec1[:,np.argmax(eig_val1)]
    vec2 = eig_vec2[:,np.argmax(eig_val2)]
    
    assert vec1.shape == vec2.shape
    
    return val1, val2, vec1, vec2

def dist(vec1, vec2):
    """
    Euclidean distance between two vectors
    """
    return np.linalg.norm(vec1-vec2)

def mahalanobis(vec, mean, cov):
    """
    Compute the mahalanobis distance of the given vector, vec
    """
    subtracted = np.subtract(vec, mean)
    return math.sqrt(np.matmul(np.matmul(subtracted.T, np.linalg.inv(cov)),subtracted))

def weight_set(means, weights, bstr_means, cov_mat, bstr=False):
    """
    Given gmm model, set the component with larger weight to be the first component and the component with
    smaller weight to be the second component
    """
    
    val1, val2, vec1, vec2 = eig(cov_mat) #eigenvalues and eigenvecotrs for covariance matrices
    
    # designate the larger component to be the component1 and the smaller component the component2 
    
    if bstr: 
        if weights[0] > weights[1]:
            mean1 = means[0]
            mean2 = means[1]
        else:
            mean1 = means[1]
            mean2 = means[0]

        if mahalanobis(bstr_means[0], mean1, cov_mat[0]) < mahalanobis(bstr_means[1], mean1, cov_mat[1]):
            return val1, val2, vec1, vec2
        else:
            return val2, val1, vec2, vec1
    else:
        if weights[0] > weights[1]:
            return val1, val2, vec1, vec2
        else:
            return val2, val1, vec2, vec1

def bstr(function, *args, n=5000):
    """
    Given data matrix, perform bootstrapping by collecting n samples (default = 5000) and return the 
    error rate for the mean of the data. Assume that the given data matrix is numpy array
    """   
    vals = [] # the primary value of our interest on which we perform bootstrapping 
         
    args = [np.asarray(i) for i in args]
    assert all(len(i) == len(args[0]) for i in args) # check all the inputs have the same length

    for i in range(n):
        
        resample = np.random.randint(0, len(args[0]), size=len(args[0]))
        resampled = [i[resample] for i in args]

        vals.append(function(resampled))
        
    return vals 

def angle_eig(resampled):
    """
    Inner function for bootstrapping on the largest eigenvalues as well as angles
    """    
    gmm_resampled = GMM(n_components=2).fit(resampled[0])
    gmm_weights = gmm_resampled.weights_
    gmm_cov = gmm_resampled.covariances_
    gmm_means = gmm_resampled.means_

    eig_val1, eig_val2, eig_vec1, eig_vec2 = weight_set(orig_means, orig_weights, gmm_means, gmm_cov, bstr=True)

    return (eig_val1, eig_val2, eig_vec1, eig_vec2, angle(eig_vec1, eig_vec2))

#fit GMM
gmm = GMM(n_components=2).fit(CC_scaled)
orig_means = gmm.means_
orig_cov = gmm.covariances_
orig_weights = gmm.weights_

P, D, Q = tailored_svd(CC_scaled)

eigval1, eigval2, eigvec1, eigvec2 = weight_set(orig_means, orig_weights, orig_means, orig_cov)
orig_angle = angle(eigvec1, eigvec2) # angle between the main eigenvectors 
orig_angle1 = angle(eigvec1, Q.T[:,0]) # angle between PC1 and the first cluster
orig_angle2 = angle(eigvec2, Q.T[:,0]) # angle between PC1 and the second cluster

values = bstr(angle_eig, CC_scaled)
perc_one, perc_two, first_eigvec, second_eigvec, bstr_angle = list(zip(*values))

angle_first = [angle(i, Q.T[:, 0]) for i in first_eigvec]
angle_second = [angle(j, Q.T[:, 0]) for j in second_eigvec]

# plot the histogram for eigenvalues and angles
num_bins = 200
itera = [perc_one, perc_two, angle_first, angle_second, bstr_angle]
names = ['largest_eignvalue1', 'largest_eigvenvalue2', 'angle1_pc1', 'angle2_pc1', 'angle_between']

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 18)
gs.update(wspace=3, hspace=0.5)
ax1 = plt.subplot(gs[0, :6])
ax2 = plt.subplot(gs[0, 6:12])
ax3 = plt.subplot(gs[0, 12:18])
ax4 = plt.subplot(gs[1, 3:9])
ax5 = plt.subplot(gs[1, 9:15])

for i in range(len(itera)):

    if i == 0:
        ax1.hist(itera[i], num_bins, facecolor='blue', alpha=0.5)
        ax1.set_title('eigenvalues of component 1')
        ax1.set_xlabel('percentage of largest eigenvalue')
        ax1.set_ylabel("number of occurences")
        ax1.axvline(x=eigval1, c= 'Black')

    elif i == 1:

        ax2.hist(itera[i], num_bins, facecolor='blue', alpha=0.5)
        ax2.set_title('eigenvalues of component 2')
        ax2.set_xlabel('percentage of largest eigenvalue')
        ax2.axvline(x=eigval2, c= 'Black')

    elif i == 2: 

        ax3.hist(itera[i], num_bins, facecolor='blue', alpha=0.5)
        ax3.set_title('Between PC1 and eigenvector from component 1')
        ax3.set_xlabel('angle')
        ax3.axvline(x=orig_angle1, c= 'Black')

    elif i == 3: 

        ax4.hist(itera[i], num_bins, facecolor='blue', alpha=0.5)
        ax4.set_title('Between PC1 and eigenvector from component 2')
        ax4.set_xlabel('angle')
        ax4.set_ylabel("number of occurences")
        ax4.axvline(x=orig_angle2, c= 'Black')

    elif i == 4:

        ax5.hist(itera[i], num_bins, facecolor='blue', alpha=0.5)
        ax5.set_title('Between eigenvectors of two Gaussians')
        ax5.set_xlabel('angle')
        ax5.axvline(x=orig_angle, c= 'Black')

plt.savefig(os.path.join('figures', "bootstrap.pdf"))
plt.close()
print("Done with bootstrap plots (Supplementary Figure 4)")

########################################################################
# Simulated histogram from Markov transition (Supplementary Figure 5, 6)
########################################################################

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

d_x_out = velArrayOut[:,:2,0] #The starting point of "OUT" vector in the first 2 PC space
d_y_out = velArrayOut[:,2:,0] #The starting point of "OUT" vector in the other 7 PC space
d_v_out = velArrayOut[:,:2,1] #The "OUT" velocity in the first 2 PC space
d_w_out = velArrayOut[:,2:,1] #The "OUT" velocity in the other 7 PC space
d_xy_out = velArrayOut[:,:,0] #The starting point of "OUT" vector in 9 PC space

pos_v_not_nan_out = np.where(~np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of non-NaN points due to end point
pos_v_nan_out = np.where(np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of NaN points due to end point

n_obs_out = len(pos_v_not_nan_out)

#Removing NaN
d_x_notnan_out = d_x_out[pos_v_not_nan_out,:]
d_xy_notnan_out = d_xy_out[pos_v_not_nan_out,:]
d_v_notnan_out = d_v_out[pos_v_not_nan_out,:]

init_PC1 = d_x_out[pos_v_nan_in,0]
final_PC1 = d_x_out[pos_v_nan_out,0]
PC1 = d_x_notnan_out[:,0]
vel_PC1 = d_v_notnan_out[:,0]

def cum_transition(PC1,vel_PC1_annual,init_PC1, years_move=100,n_bin=6,n_iter=10 ,flag_barrier=False, graph=True,
					graph_name='' ,flag_rm_jump=False, transition_prob_input=None,ratio_init_input=None):
    vel_PC1 = vel_PC1_annual*years_move
    left_end = np.min(PC1)#np.min(PC1+vel_PC1)
    right_end = np.max(PC1)# np.max(PC1+vel_PC1)
    width = (right_end-left_end)/n_bin
    center_list =( np.linspace(left_end,right_end-width,n_bin) + np.linspace(left_end+width,right_end,n_bin) )/2

    if ratio_init_input is None:
        count_init_PC1 = np.zeros(n_bin)
        for i in range(n_bin):
            loc = (left_end+i*width <=init_PC1)*(init_PC1<left_end+(i+1)*width)
            count_init_PC1[i] = np.sum(loc)
        ratio_init = count_init_PC1/np.sum(count_init_PC1)
        
    else:
        ratio_init = ratio_init_input
        
    if transition_prob_input is None:
        transition_count_matrix = np.zeros([n_bin,n_bin])
        transition_prob_matrix = np.zeros([n_bin,n_bin])
        for i in range(n_bin):
            loc_origin = (left_end+i*width <=PC1)*(PC1<left_end+(i+1)*width)
            num_origin_i = np.sum(loc_origin)
            if num_origin_i == 0:
                print('No observation starting in bin %i'%i)
            center = left_end+(i+.5)*width
            dest = center+vel_PC1[loc_origin]
            if flag_barrier:
                loc_dest_right = (dest>right_end)
                transition_count_matrix[i,-1]=transition_count_matrix[i,-1]+np.sum(loc_dest_right)
                loc_dest_left = (dest<left_end)
                transition_count_matrix[i,0]=transition_count_matrix[i,0]+np.sum(loc_dest_left)
            for j in range(n_bin):
                loc_dest_j = (left_end+j*width <=dest)*(dest<left_end+(j+1)*width)
                transition_count_matrix[i,j] = transition_count_matrix[i,j] + np.sum(loc_dest_j)
                if flag_rm_jump:
                    transition_count_matrix[:5,7:] = 0
                
                transition_prob_matrix[i,j] = transition_count_matrix[i,j]/num_origin_i
    else:
        transition_count_matrix = None
        transition_prob_matrix = transition_prob_input
            
    #transition_count_matrix,transition_prob_matrix = calc_transition_matrix(PC1,vel_PC1,n_bin)
    #---------
    
    dist_transition = []  
    dist_cum_transition = []      

    dist_i = ratio_init.reshape([-1,1])
    dist_cum_i = ratio_init.reshape([-1,1])
        
    for i in range(n_iter):
        dist_transition.append(dist_i)
        dist_cum_transition.append(dist_cum_i)
        dist_i = np.matmul(transition_prob_matrix.T,dist_i)
        dist_cum_i = dist_cum_i + dist_i
            
    return transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition
        

def nga_specific_iteration_length(years_move, n_bin=10, n_iter=20, flag_barrier=True, flag_rm_jump=False):
    transition_count_matrix_b, transition_prob_matrix_b,ratio_init,_,_ = cum_transition(PC1,vel_PC1,init_PC1, years_move=years_move, n_bin=n_bin,
    																					n_iter=n_iter,flag_barrier=flag_barrier,flag_rm_jump=flag_rm_jump )    
    #transition_count_matrix, transition_prob_matrix,ratio_init = cum_transition(PC1,vel_PC1,init_PC1, years_move=500,n_bin=10,n_iter=15,flag_barrier=False  )    

    end_year_NGA = np.array(flowInfo.loc[pos_v_nan_out]['Time']).astype(float)
    start_year_NGA =np.array(flowInfo.loc[pos_v_nan_in]['Time']).astype(float) 
    duration_NGA = end_year_NGA-start_year_NGA
    n_iter_list =   np.floor(duration_NGA/years_move).astype(int)

    left_end = np.min(PC1)#np.min(PC1+vel_PC1)
    right_end = np.max(PC1)# np.max(PC1+vel_PC1)
    width = (right_end-left_end)/n_bin
    center_list =( np.linspace(left_end,right_end-width,n_bin) + np.linspace(left_end+width,right_end,n_bin) )/2


    init_position_list = []
    dist_transition_list = []
    dist_cum_transition_list = []
    for i in range(30):
        init = init_PC1[i]
        init_PC1_NGA = np.zeros(n_bin)
        for j in range(n_bin):
            init_PC1_NGA[j] = np.sum( (left_end+j*width <=init) * (init<left_end+(j+1)*width))
            
        init_position_list.append(init_PC1_NGA)
        if n_iter_list[i]>0:
            transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter_list[i],flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False )    
        else:
            dist_transition = init_PC1_NGA
            dist_cum_transition = init_PC1_NGA
        dist_transition_list.append(dist_transition)
        dist_cum_transition_list.append(dist_cum_transition)        

    dist_sum_NGA = np.zeros(n_bin)
    for i in range(30):
        dist_cum_final = np.sum(dist_transition_list[i],axis=0)
        dist_sum_NGA = (dist_sum_NGA +dist_cum_final.flatten() )

    if years_move == 100:
        plt.bar(center_list, dist_sum_NGA, color='r')
    else:
        plt.bar(center_list, dist_sum_NGA, color='g')

    plt.savefig(os.path.join('figures', "simulated-histogram-transition-matrix-NGA-specific-length-%iyears.pdf"%(years_move)))
    plt.close()

# Figure 6a
nga_specific_iteration_length(500)

n_bin=10
years_move=500

left_end = np.min(PC1)#np.min(PC1+vel_PC1)
right_end = np.max(PC1)# np.max(PC1+vel_PC1)
width = (right_end-left_end)/n_bin
center_list =( np.linspace(left_end,right_end-width,n_bin) + np.linspace(left_end+width,right_end,n_bin) )/2

end_year_NGA = np.array(flowInfo.loc[pos_v_nan_out]['Time']).astype(float)
start_year_NGA =np.array(flowInfo.loc[pos_v_nan_in]['Time']).astype(float) 
duration_NGA = end_year_NGA-start_year_NGA
n_iter_list =   np.floor(duration_NGA/years_move).astype(int)

flag_barrier=1
dist_transition_from_unif_list = []
dist_cum_transition_from_unif_list = []
ratio_init_input = np.ones(n_bin)/n_bin
n_iter=10
for i in range(30):
    init = init_PC1[i] #doesn't matter
    
    transition_count_matrix_from_unif, transition_prob_matrix_from_unif,ratio_init,dist_transition_from_unif,dist_cum_transition_from_unif = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter,flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False,ratio_init_input=ratio_init_input )    
    dist_transition_from_unif_list.append(dist_transition_from_unif)
    dist_cum_transition_from_unif_list.append(dist_cum_transition_from_unif)    

dist_sum_NGA_from_unif = np.zeros(n_bin)
for i in range(30):
    dist_cum_final_from_unif = np.sum(dist_transition_from_unif_list[i],axis=0)
    dist_sum_NGA_from_unif = (dist_sum_NGA_from_unif +dist_cum_final_from_unif.flatten() )

# Figure 6e
plt.bar(np.array(center_list), dist_sum_NGA_from_unif, color='y')
plt.savefig(os.path.join('figures', 
	"simulated-histogram-starting-from-uniform-dist(iter%i-bins%i-timescale%iyears-barrier%i).pdf"%(n_iter, n_bin,years_move,flag_barrier)))
plt.close()

plt.bar(center_list[:n_bin-1], dist_sum_NGA_from_unif[:n_bin-1], color='y')
plt.savefig(os.path.join('figures', 
	"simulated-histogram-starting-from-uniform-dist-no-last-bin(iter%i-bins%i-timescale%iyears-barrier%i).pdf"%(n_iter, n_bin,years_move,flag_barrier)))
plt.close()

#%%
'''
Controlling mean and/or variance
'''
from scipy.optimize import minimize
from scipy.special import kl_div



dist_transition_list = []
dist_cum_transition_list = []
ratio_init_input = np.ones(n_bin)/n_bin
n_iter=10

for i in range(30):
    init = init_PC1[i] #doesn't matter
    
    transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter,flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False,ratio_init_input=None )    
    dist_transition_list.append(dist_transition)
    dist_cum_transition_list.append(dist_cum_transition)

dist_sum_NGA = np.zeros(n_bin)
for i in range(30):
    dist_cum_final = np.sum(dist_transition_list[i],axis=0)
    dist_sum_NGA = (dist_sum_NGA +dist_cum_final.flatten() )
# plt.bar(np.array(center_list), dist_sum_NGA, color='y')

'''
KL divergence of original and modified transition, controlling mean and variance.
To be minimzied.
'''
def func_obj(p, *args):
    q,x,mu,sig2,pos_to_empty = args
    
    q_nonzero = q[np.nonzero(q)]
    p_nonzero = p[np.nonzero(q)]
    
    
    kl = kl_div(p_nonzero,q_nonzero)
    '''
    '''
    con0 = np.abs( np.sum(p)-1. )
    con1= np.abs(np.sum(p*x)-mu )
    con2= np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 )
    
    cons3 = np.sum( np.abs(p[ np.where( q==0 ) ] ) )
    
    cons4 = np.sum( np.abs(p[pos_to_empty] ) )
    
    
    return np.sum(kl) 

'''
Simulate transition with modified transition matrix
'''    
def sim_trans(transition_prob_matrix,figtitle=None,filename=None, color='g'):
    init_position_list = []
    dist_transition_list = []
    dist_cum_transition_list = []
    for i in range(30):
        init = init_PC1[i]
        init_PC1_NGA = np.zeros(n_bin)
        for j in range(n_bin):
            init_PC1_NGA[j] = np.sum( (left_end+j*width <=init) * (init<left_end+(j+1)*width))
            
        init_position_list.append(init_PC1_NGA)
        if n_iter_list[i]>0:
            temp1, temp2,ratio_init,dist_transition,dist_cum_transition = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter_list[i],flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False, transition_prob_input = transition_prob_matrix_modified )    
        else:
            dist_transition = init_PC1_NGA
            dist_cum_transition = init_PC1_NGA
        dist_transition_list.append(dist_transition)
        dist_cum_transition_list.append(dist_cum_transition)
        
    
    dist_sum_NGA = np.zeros(n_bin)
    for i in range(30):
        dist_cum_final = np.sum(dist_transition_list[i],axis=0)
        dist_sum_NGA = (dist_sum_NGA +dist_cum_final.flatten() )
    plt.bar(center_list, dist_sum_NGA, color=color)
    # plt.title(figtitle)
    plt.savefig(os.path.join('figures', filename))
    plt.close()

'''
Match the mean and/or variance of the transition starting from each bin (2nd to 8th).
Minimize KL divergence
'''

a=np.arange(n_bin).reshape([1,-1] ) 
aa = a
for i in range(1,n_bin):
    aa=np.concatenate( (aa,a-i),axis=0 )
    
mean_step = np.sum(transition_prob_matrix*aa , axis=1)
var_step = np.sum(transition_prob_matrix*np.power(aa,2) , axis=1) - np.power(mean_step,2)

mean_all = np.sum(transition_count_matrix*aa )/np.sum(transition_count_matrix)
sig2_all = np.sum(transition_count_matrix*np.power(aa,2) )/np.sum(transition_count_matrix) - np.power(mean_all,2)

#Match both mean and variance
transition_prob_matrix_modified = np.copy(transition_prob_matrix)
bnds_temp = np.array([[0.,1.]]*10 )
for i in range(1,8):
    q = np.copy(transition_prob_matrix[i,:] )
    #p_init = np.copy(q)
    
    p_init = np.zeros(10)
    p_init[np.where(q!=0)] = 1/len(np.where(q!=0)[0])
    
    x=aa[i,:]
    mu = mean_all
    sig2=sig2_all
    args = (q,x,mu,sig2,[]) 
    bnds = np.copy(bnds_temp)
    bnds[np.where(q==0)] = [0.,0.]
    cons=({'type':'eq','fun':lambda p: np.abs( np.sum(p)-1. )  } , {'type':'eq','fun':lambda p: np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) }, {'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }  )
    
    #,{'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }
    sol = minimize(func_obj, p_init, args, method = 'SLSQP',options={'ftol':10e-10,'maxiter':20000},bounds=bnds,constraints=cons)
    p =sol.x
    # print('---%i---'%i)
    # print('prob:',p)
    # print( 'sum of prob:',np.sum(p) )
    # print( 'error in mean:',np.abs(np.sum(p*x)-mu ) )
    # print( 'error in var:',np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
    # print(  'pos q 0:',np.abs(p[ np.where( q==0 ) ] ) )
    
    transition_prob_matrix_modified[i,:] = p

# # Figure 6d
sim_trans(transition_prob_matrix_modified,figtitle='Simulated histogram, fixed mean and variance in velocity',filename='simulated-histogram-fixed-mean-and-var.pdf', color='b')

#Match  var only
transition_prob_matrix_modified = np.copy(transition_prob_matrix)
bnds_temp = np.array([[0.,1.]]*10 )
for i in range(1,8):
    q = np.copy(transition_prob_matrix[i,:] )
    #p_init = np.copy(q)
    
    p_init = np.zeros(10)
    p_init[np.where(q!=0)] = 1/len(np.where(q!=0)[0])
    
    x=aa[i,:]
    mu = mean_all
    sig2=sig2_all
    args = (q,x,mu,sig2,[]) 
    bnds = np.copy(bnds_temp)
    bnds[np.where(q==0)] = [0.,0.]
    cons=({'type':'eq','fun':lambda p: np.abs( np.sum(p)-1. )  } , {'type':'eq','fun':lambda p: np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) }  )
    #,{'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }
    sol = minimize(func_obj, p_init, args, method = 'SLSQP',options={'ftol':10e-10,'maxiter':20000},bounds=bnds,constraints=cons)
    p =sol.x
    # print('---%i---'%i)
    # print('prob:',p)
    # print( 'sum of prob:',np.sum(p) )
    # print( 'error in mean:',np.abs(np.sum(p*x)-mu ) )
    # print( 'error in var:',np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
    # print(  'pos q 0:',np.abs(p[ np.where( q==0 ) ] ) )
    
    transition_prob_matrix_modified[i,:] = p

# Figure 6c
sim_trans(transition_prob_matrix_modified,figtitle='Simulated histogram, fixed variance in velocity',filename='simulated-histogram-fixed-var.pdf', color='c')

#Match  mean only
transition_prob_matrix_modified = np.copy(transition_prob_matrix)
bnds_temp = np.array([[0.,1.]]*10 )
for i in range(1,8):
    q = np.copy(transition_prob_matrix[i,:] )
    #p_init = np.copy(q)
    
    p_init = np.zeros(10)
    p_init[np.where(q!=0)] = 1/len(np.where(q!=0)[0])
    
    x=aa[i,:]
    mu = mean_all
    sig2=sig2_all
    args = (q,x,mu,sig2,[]) 
    bnds = np.copy(bnds_temp)
    bnds[np.where(q==0)] = [0.,0.]
    cons=({'type':'eq','fun':lambda p: np.abs( np.sum(p)-1. )  } , {'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) } )
    #,{'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }
    sol = minimize(func_obj, p_init, args, method = 'SLSQP',options={'ftol':10e-10,'maxiter':20000},bounds=bnds,constraints=cons)
    p =sol.x
    # print('---%i---'%i)
    # print('prob:',p)
    # print( 'sum of prob:',np.sum(p) )
    # print( 'error in mean:',np.abs(np.sum(p*x)-mu ) )
    # print( 'error in var:',np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
    # print(  'pos q 0:',np.abs(p[ np.where( q==0 ) ] ) )
    
    transition_prob_matrix_modified[i,:] = p

# Figure 6b
sim_trans(transition_prob_matrix_modified,figtitle='Simulated histogram, fixed mean in velocity',filename='simulated-histogram-fixed-mean.pdf', color='m')
print("Done with simulation histogram (Supplementary Figure 6)")

# Figure 5
plt.imshow(transition_prob_matrix, cmap='hot')
plt.xticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.yticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.colorbar()
plt.savefig(os.path.join('figures', "heatmap-of-transition-matrix.pdf"))
plt.close()
print("Done with heatmap (Supplementary Figure 5)")

###############################################################################
# Simulation models (Supplementary Figure 7, 8)
###############################################################################

def simulation_plots(L1, k1, x_start1, L2, k2, x_start2, figure):
    """
    Generate simulation plots for Figure 10 and 11
    """
    x = -np.random.chisquare(2,4000)
    x_lin = np.linspace(x.min(),x.max(),num=4000)

    y1 = L1/(1.+np.exp(-k1*(x-x_start1 )))
    y1_lin = L1/(1.+np.exp(-k1*(x_lin-x_start1 )))

    f, axes = plt.subplots(2, 2, figsize=(12,12))
    axes[0, 0].plot(x_lin, y1_lin)
    axes[0, 0].set_title("First CC growth")

    y2 = L2/(1.+np.exp(-k2*(x-x_start2 )))
    y2_lin = L2/(1.+np.exp(-k2*(x_lin-x_start2 )))

    axes[0, 1].plot(x_lin, y2_lin)
    axes[0, 1].set_title("Second CC growth")

    Y=.5*y1+.5*y2
    Y_lin=.5*y1_lin+.5*y2_lin
    axes[1,0].plot(x_lin, Y_lin)
    axes[1,0].set_title("Growth of the average")

    n, bins, patches = plt.hist(Y, 200,  facecolor='blue', alpha=0.5)
    plt.title('Histogram of the average of two CC')
    axes[1, 1].hist(Y, 200, facecolor='blue', alpha=0.5)
    axes[1, 1].set_title('Histogram of the average of two CC')

    for ax in axes.flat:    
        ax.set(xlabel='time', ylabel='CC value')

    for ax in axes.flat:
        ax.xaxis.get_label().set_fontsize(12)
        ax.yaxis.get_label().set_fontsize(12)
        ax.label_outer()

    if figure==10:
        plt.savefig(os.path.join('figures', 'S1.png'))
    else:
        plt.savefig(os.path.join('figures', 'S2.png'))

    plt.clf()
    plt.close()

#Figure 7
simulation_plots(1, 1, -10, 5, 1, -1, 10)
print(f"Done with simulation models (Supplementary Figure 7)")

#Figure 8
simulation_plots(1, 1, -10, 5, 1, -5, 11)
print(f"Done with simulation models (Supplementary Figure 8)")

###############################################################################
# Interpolation plots (Supplementary Figure 9)
###############################################################################

dataPath1 = os.path.abspath(os.path.join("./..","Seshat_arxiv","data1.csv")) #20 imputed sets
dataPath2 = os.path.abspath(os.path.join("./..","Seshat_arxiv","data2.csv")) #Turchin's PCs

pnas_data1 = pd.read_csv(dataPath1)
pnas_data2 = pd.read_csv(dataPath2)

num_bins = 200

# interpolating on the 1-d principal axis
period = [pnas_data1.loc[i]['Time'] for i in range(len(PC_matrix[:,0]))]
added_rows = list()

for impute_step in range(1, 21):
    # 1) polity-based interpolation
    impute_set = pnas_data1[pnas_data1.irep==impute_step]
    unique_region = impute_set.NGA.unique().tolist()
    

    for nga in unique_region:
        times = sorted(impute_set[impute_set.NGA == nga].Time.unique().tolist())
        if len(times) != ((max(times)-min(times))/100)+1:
            for time in range(len(times)-1):
                if times[time+1]-times[time] != 100:
                    # linear interpolation
                    val1_idx = pnas_data1.index[(pnas_data1['NGA'] == nga) & 
                                                (pnas_data1['Time'] == times[time]) &
                                                (pnas_data1['irep'] == impute_step)].tolist()[0]
                    val2_idx = pnas_data1.index[(pnas_data1['NGA'] == nga) & 
                                                (pnas_data1['Time'] == times[time+1]) &
                                               (pnas_data1['irep'] == impute_step)].tolist()[0]

                    diff = PC_matrix[:,0][val2_idx] - PC_matrix[:,0][val1_idx]
                    
                    num_steps = int((times[time+1]-times[time])/100)

                    for i in range(1, num_steps):
                        diff_step = (i/num_steps)*diff
                        interpol = diff_step+PC_matrix[:,0][val1_idx]
                        added_rows.append(interpol)
                        period.append(times[time]+100*i)

added_rows = np.asarray(added_rows)

pc_proj = np.concatenate((PC_matrix[:,0], added_rows))

assert len(pc_proj) == len(period) 
    

# compute the average temporal gap size for each data point (with respect to regions)
temporal_gap = dict()
weighted = list()
unique_region = pnas_data1.NGA.unique().tolist()
period = list()

for impute_step in range(1, 21):
    for nga in unique_region:
        times = sorted(pnas_data1[pnas_data1.NGA == nga].Time.unique().tolist())
        for time in range(len(times)):
            idx = pnas_data1.index[(pnas_data1.irep == impute_step) & (pnas_data1.NGA == nga) & 
                                    (pnas_data1.Time == times[time])].tolist()[0]
            if time == 0: # the data point is the first appearance of a polity in the nga
                temporal_gap[idx] = int(round((times[time+1]-times[time])))
            elif time == len(times)-1: # the data point is the last time period of the recorded nga
                temporal_gap[idx] = int(round((times[time] - times[time-1])))
            else: 
                temporal_gap[idx] = int(round((-times[time-1]+times[time+1])/2))

assert len(temporal_gap.keys()) == 8280 # temporal_gap must contain all 8280 points

# wegiht the occurence of each data point by their respective temporal gap size
for i in range(len(PC_matrix[:,0])):
    weighted.extend(temporal_gap[i]*[PC_matrix[:,0][i]])

#fit GMM
gmm = GMM(n_components=2)
gmm = gmm.fit(X=np.expand_dims(PC_matrix[:,0], 1))

# mean and covariance for each component
gauss_one = gmm.weights_[0] #weight for gaussian distribution
gauss_two = gmm.weights_[1] #weight for gaussian distribution 

idx_data = sorted(range(len(PC_matrix[:,0])), key = lambda i: PC_matrix[:,0][i]) #sort data points by their values on principal axis
prob_distr = gmm.predict_proba(X=np.expand_dims(sorted(PC_matrix[:,0]), 1)) #probability distribution for gaussians on principal axis
lower = list()
higher = list()

for idx in range(len(prob_distr)):
    if gmm.weights_[0] < gmm.weights_[1]:
        if prob_distr[idx][0] >= prob_distr[idx][1]:
            lower.append(idx_data[idx])
        else:
            higher.append(idx_data[idx])
    else:
        if prob_distr[idx][0] <= prob_distr[idx][1]:
            lower.append(idx_data[idx])
        else:
            higher.append(idx_data[idx])

both_ngas = list()
# return polities that lie in both Gaussians for each imputed set
for i in range(1,21):
    impute = pnas_data1[pnas_data1.irep == i]
    lower_df = impute.loc[lower].dropna()
    higher_df = impute.loc[higher].dropna()

    unique_nga = [j for j in impute.NGA.unique().tolist() if (j in lower_df.NGA.unique().tolist()
                                                         and j in higher_df.NGA.unique().tolist())]
    nga_gauss = impute.loc[pnas_data1['NGA'].isin(unique_nga)]
    both_gauss = np.take(PC_matrix[:,0], nga_gauss.index.values)
    both_ngas.append(both_gauss)

for i in range(len(both_ngas)-1):
    ngas = np.concatenate((both_ngas[i], both_ngas[i+1]), axis=0)
    both_ngas[i+1] = ngas

gs = gridspec.GridSpec(3,1)

fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(gs[0])
ax.hist(pc_proj, num_bins, normed=1, facecolor='blue', alpha=0.5)

ax = fig.add_subplot(gs[1])
ax.hist(weighted, num_bins, normed=1, facecolor='blue', alpha=0.5)

ax = fig.add_subplot(gs[2])
ax.hist(ngas, num_bins, normed=1, facecolor='blue', alpha=0.5)

fig.text(0.5, 0.04, 'first PC value', ha='center')
fig.text(0.02, 0.5, 'normalized probability density', va='center', rotation='vertical')

fig.savefig(os.path.join('figures', 'validation_interpolation.pdf'))
plt.close()

print("Done with interpolation plots (Supplementary Figure 9)")

#########################################################################################################
# Create a 5 x 2 plot to show time sequences organized by the ten world regions (Supplementary Figure 10)
#########################################################################################################

f, axes = plt.subplots(int(len(worldRegions)/2),2, sharex=True, sharey=True,figsize=(12,15))
axes[0,0].set_xlim([t_min,t_max])
axes[0,0].set_ylim([pc1_min,pc1_max])
for i,reg in enumerate(worldRegions):
    regList = list(reversed(regionDict[reg]))
    # m,n index the subplots
    m = i % int(len(worldRegions)/2) # mod; result is 0, 1, 2, 3, or 4
    n = i // int(len(worldRegions)/2) # integer division; result is 0 or 1
    for nga in regList:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        pc1 = list()
        for t in times:
            ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
            pc1.append(np.mean(PC_matrix[ind,0]))
        axes[m,n].scatter(times,pc1,s=10)
    s = '{' + regList[0] + '; ' + regList[1] + '; ' + regList[2] + '}'
    axes[m,n].set_title(s,fontsize=10)
    if m != 4:
        plt.setp(axes[m,n].get_xticklabels(), visible=False)
    else:
        axes[m,n].set_xlabel("Calendar Date [AD]")
    if n == 0:
        axes[m,n].set_ylabel("PC1")

f.subplots_adjust(hspace=.5)
plt.savefig(os.path.join('figures', "pc1_vs_time_stacked_by_region.pdf"))
plt.close()
print("Done with time stacked region plot (Supplementary Figure 10)")

print("All figures generated")
