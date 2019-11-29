import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, PolynomialFeatures
from sklearn.mixture import GaussianMixture as GMM
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal, norm

from scipy.interpolate import SmoothBivariateSpline
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

import time
import pickle
import os

import tensorflow as tf
import tensorflow_probability as tfp

from collections import defaultdict
import os
import math
from sklearn.decomposition import PCA, FastICA
from sklearn import linear_model
import scipy.spatial as spatial
from IPython.core.debugger import Pdb

from multiprocessing import Pool
import multiprocessing as mp
# Define some functions
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)

def tailored_svd(data):
    # perform singular value decomposition on the given data matrix
    #center the data
    mean = np.mean(data, axis=0)
    data -= mean
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    return P, D, Q

# Create the grid for PC1 and PC2 in a standalone file to avoid replicating code
def createGridForPC12(dGrid,flowArray):
    # Remove endpoints
    ind = [True if not np.isnan(flowArray[i,0,1]) else False for i in range(flowArray.shape[0])]
    fa = flowArray[ind,:,:]
    points2D = fa[:,range(0,2),0]

    u0Min = np.floor(np.min(points2D[:,0] - dGrid) / dGrid) * dGrid # PC1 min
    u0Max = np.ceil(np.max(points2D[:,0] + dGrid) / dGrid) * dGrid # PC1 max
    v0Min = np.floor(np.min(points2D[:,1] - dGrid) / dGrid) * dGrid # PC1 min
    v0Max = np.ceil(np.max(points2D[:,1] + dGrid) / dGrid) * dGrid # PC1 max
    u0Vect = np.arange(u0Min,u0Max,dGrid)
    v0Vect = np.arange(v0Min,v0Max,dGrid)
    return u0Vect,v0Vect

#Vectorize samples
def Ev_given_x_ygrid2(x_grid, y_samples,sigma2):
    n_samples = y_samples.shape[0]
    n_obs = len(pos_v_not_nan)
    x_grid_ex = np.repeat(x_grid.reshape([1,-1]),n_samples*n_obs ,axis=0)
    y_samples_ex = np.tile(y_samples,(n_obs,1))
    xy_ex = np.concatenate((x_grid_ex,y_samples_ex),axis=1)
    d_xy_ex = np.repeat(d_xy[pos_v_not_nan,:],n_samples,axis=0)
    d_v_ex = np.repeat(d_v[pos_v_not_nan,:],n_samples,axis=0)
    phi = np.prod( norm.pdf(d_xy_ex,loc=xy_ex,scale=np.power(sigma2,.5)),axis=1)
    phi2 = np.repeat( phi.reshape([-1,1]), 2, axis=1 )

    v_weighted = d_v_ex * phi2

    num = np.sum(v_weighted,axis=0)
    denom = np.sum(phi2,axis=0)

    velocity_est = num/denom

    return velocity_est

#TF version
def Ev_given_xy_tf(x_grid, y_sample,sigma2):
    m = tf.concat([x_grid,y_sample],axis=0)
    mvn = tfd.MultivariateNormalDiag(    loc=m,    scale_diag=sigma2) #sigma2 is (9,)
    phi_seq = mvn.prob(d_xy_tf)
    phi_seq2 = tf.tile(tf.reshape(phi_seq,[-1,1]),[1,2])
    s = d_v_tf*phi_seq2
    num = tf.reduce_sum(s ,axis=0)
    denom = tf.reduce_sum(phi_seq)
    velocity = num/denom

    return velocity

def Ev_given_x_ygrid_tf(x_grid, y_samples_tf,n_samples,sigma2):

    velocity_each_sample = np.zeros( [n_samples, 2] )

    for i,y_sample in enumerate(y_samples_tf):
        velocity_each_sample[i,:] = Ev_given_xy_tf(x_grid, y_sample,sigma2)
    velocity_est = np.mean(velocity_each_sample,axis=0)

    vel_sum = tf.zeros(shape=[2])
    for i in range(n_samples):
        y_sample = y_samples_tf[i,:]
        vel = Ev_given_xy_tf(x_grid, y_sample,sigma2)
        vel_sum = vel_sum+vel
    velocity_est = vel_sum/n_samples

    return velocity_est

def Ev_given_x_ygrid_tf2(x_grid, y_samples_tf,n_samples,sigma2):

    x_ex = tf.tile(tf.reshape(x_grid,[1,-1]), [n_obs,1])
    x_ex2 = tf.concat( (x_ex,  tf.zeros([n_obes,7])), axis=1 )
    d_xy_temp = d_xy_tf - x_ex2
    d_xy_temp2 = tf.tile( d_xy_temp,[n_samples,1]  )
    y_ex = tf.tile( y_samples_tf, [n_obs,1] )

def calcMeanVectorsSlice(u0Vect,v0Vect,r0,flowArray,minPoints=10):
    # Remove endpoints
    ind = [True if not np.isnan(flowArray[i,0,1]) else False for i in range(flowArray.shape[0])]
    fa = flowArray[ind,:,:]

    points9D = flowArray[:,:,0]
    points9D = points9D[ind,:]
    tree9D = spatial.cKDTree(points9D)

    # Calculate the mean for the remaining seven PCs
    otherMeans = np.empty(7)
    for ii in range(0,flowArray.shape[1]-2):
        otherMeans[ii] = np.mean(flowArray[:,ii+2,0])

    # Initialize the matrices
    du0Mat = np.empty((len(u0Vect),len(v0Vect)))
    dv0Mat = np.empty((len(u0Vect),len(v0Vect)))
    mm0Mat = np.empty((len(u0Vect),len(v0Vect)))

    for i,u0 in enumerate(u0Vect):
        for j,v0 in enumerate(v0Vect):
            z0 = np.hstack((u0,v0,otherMeans))
            neighb = tree9D.query_ball_point(z0,r0)
            if len(neighb) >= minPoints:
                du0Mat[i,j] = np.mean(fa[neighb,0,1])
                dv0Mat[i,j] = np.mean(fa[neighb,1,1])
                #Pdb().set_trace()
                mm0Mat[i,j] = np.sqrt(np.sum(np.power(fa[neighb,0,1] - du0Mat[i,j],2) + np.power(fa[neighb,1,1]-dv0Mat[i,j],2))/len(neighb))
            else:
                du0Mat[i,j] = np.nan
                dv0Mat[i,j] = np.nan
                mm0Mat[i,j] = np.nan
    return du0Mat,dv0Mat,mm0Mat

def calcMeanVectorsWeighted(u0Vect,v0Vect,r0,flowArray,minPoints=10):
    # Using the weighting procedure describe in Supplementary Information
    # dGrid is grid spacing (same for both dimensions, PC1 and PC2)
    # r0 is radius for choosing points

    # Remove endpoints
    ind = [True if not np.isnan(flowArray[i,0,1]) else False for i in range(flowArray.shape[0])]
    fa = flowArray[ind,:,:]
    points2D = fa[:,range(0,2),0]
    tree2D = spatial.cKDTree(points2D)

    du0Mat = np.empty((len(u0Vect),len(v0Vect)))
    dv0Mat = np.empty((len(u0Vect),len(v0Vect)))
    mm0Mat = np.empty((len(u0Vect),len(v0Vect)))

    for i,u0 in enumerate(u0Vect):
        for j,v0 in enumerate(v0Vect):
            neighb = tree2D.query_ball_point([u0,v0],r0)
            if len(neighb) >= minPoints:
                weights = np.empty(len(neighb))
                for k,n in enumerate(neighb):
                    dx = np.sqrt(np.power(points2D[n,0] - u0,2) + np.power(points2D[n,1] - v0,2))
                    d = np.sqrt(np.power(r0,2) - np.power(dx,2))
                    weights[k] = np.power(d,7)
                weights = weights / np.sum(weights)
                #weights = np.ones(len(neighb)) # Uncomment to compare with old approach
                #weights = weights / len(weights)
                du0Mat[i,j] = np.sum(fa[neighb,0,1] * weights)
                #Pdb().set_trace()
                dv0Mat[i,j] = np.sum(fa[neighb,1,1] * weights)
                mm0Mat[i,j] = np.sqrt(np.sum(weights*np.power(fa[neighb,0,1] - du0Mat[i,j],2) + weights*np.power(fa[neighb,1,1]-dv0Mat[i,j],2)))
            else:
                du0Mat[i,j] = np.nan
                dv0Mat[i,j] = np.nan
                mm0Mat[i,j] = np.nan
    return du0Mat,dv0Mat,mm0Mat

def generate_core_data():
    # Read csv data files
    CC_file = "pnas_data1.csv" #20 imputed sets
    PC1_file = "pnas_data2.csv" #Turchin's PC1s
    #polity_file = os.path.abspath(os.path.join("./..","data","scraped_seshat.csv")) #Info on polities spans and gaps
    CC_df = pd.read_csv(CC_file) # A pandas dataframe
    PC1_df = pd.read_csv(PC1_file) # A pandas dataframe
#polity_df = pd.read_csv(polity_file) # A pandas dataframe

    # Create a dictionary that maps from World Region to Late, Intermediate, and Early NGAs
    regionDict = {"Africa":["Ghanaian Coast","Niger Inland Delta","Upper Egypt"]}
    regionDict["Europe"] = ["Iceland","Paris Basin","Latium"]
    regionDict["Central Eurasia"] = ["Lena River Valley","Orkhon Valley","Sogdiana"]
    regionDict["Southwest Asia"] = ["Yemeni Coastal Plain","Konya Plain","Susiana"]
    regionDict["South Asia"] = ["Garo Hills","Deccan","Kachi Plain"]
    regionDict["Southeast Asia"] = ["Kapuasi Basin","Central Java","Cambodian Basin"]
    regionDict["East Asia"] = ["Southern China Hills","Kansai","Middle Yellow River Valley"]
    regionDict["North America"] = ["Finger Lakes","Cahokia","Valley of Oaxaca"]
    regionDict["South America"] = ["Lowland Andes","North Colombia","Cuzco"]
    regionDict["Oceania-Australia"] = ["Oro PNG","Chuuk Islands","Big Island Hawaii"]

    worldRegions = list(regionDict.keys()) # List of world regions

    # Define some plotting parameters
    t_min = -10000
    t_max = 2000
    pc1_min = -7
    pc1_max = 7
    pc2_min = -7
    pc2_max = 7

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Do the singular value decomposition
    # Subset only the 9 CCs and convert to a numpy array
    CC_names = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']
    CC_array = CC_df.loc[:, CC_names].values

    # Normalize the data (across 20 imputations, not within each imputation)
    CC_scaled = StandardScaler().fit_transform(CC_array)
    CC_times = CC_df.loc[:, ['Time']].values

    # Do a singular value decomposition
    P, D, Q = tailored_svd(CC_scaled)

    # For each polity, project onto the principle components
    # PC_matrix is 8280 x 9 = (414*20) x 9
    PC_matrix = np.matmul(CC_scaled, Q.T)

    NGAs = CC_df.NGA.unique().tolist() # list of unique NGAs from the dataset

    # Create the data for the flow analysis. The inputs for this data creation are
    # the complexity characteristic dataframe, CC_df [8280 x 13], and the matrix of
    # principal component projections, PC_matrix [8280 x 9]. Each row is an imputed
    # observation for 8280 / 20 = 414 unique polity configurations. CC_df provides
    # key information for each observation, such as NGA and Time.
    #
    # Four arrays are created: movArrayOut, velArrayIn, movArrayIn, and velArrayIn.
    # All four arrays have the dimensions 414 x 9 x 2. mov stands for movements and
    # vel for velocity. 414 is the numbers of observations, 9 is the number of PCs,
    # and the final axis has two elements: (a) the PC value and (b) the change in
    # the PC value going to the next point in the NGA's time sequence (or, for vel,
    # the change divided by the time difference). The "Out" arrays give the
    # movement (or velocity) away from a point and the "In" arrays give the
    # movement (or velocity) towards a point. The difference is set to NA for the
    # last point in each "Out" sequence and the first point in each "In" sequence.
    # In addition, NGA name and time are stored in the dataframe flowInfo (the needed
    # "supporting" info for each  observation).

    # Generate the "Out" datasets
    movArrayOut = np.empty(shape=(0,9,2)) # Initialize the movement array "Out"
    velArrayOut = np.empty(shape=(0,9,2)) # Initialize the velocity array "Out" [movement / duration]
    flowInfo = pd.DataFrame(columns=['NGA','Time']) # Initialize the info dataframe

    # Iterate over NGAs to populate movArrayOut, velArrayOut, and flowInfo
    for nga in NGAs:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        for i_t,t in enumerate(times):
            ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
            newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
            flowInfo = flowInfo.append(newInfoRow,ignore_index=True)
            newArrayEntryMov = np.empty(shape=(1,9,2))
            newArrayEntryVel = np.empty(shape=(1,9,2))
            for p in range(movArrayOut.shape[1]):
                newArrayEntryMov[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                newArrayEntryVel[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                if i_t < len(times) - 1:
                    nextTime = times[i_t + 1]
                    nextInd = indNga & (CC_df['Time']==nextTime) # boolean vector for slicing also by time
                    nextVal = np.mean(PC_matrix[nextInd,p])
                    newArrayEntryMov[0,p,1] = nextVal - newArrayEntryMov[0,p,0]
                    newArrayEntryVel[0,p,1] = newArrayEntryMov[0,p,1]/(nextTime-t)
                else:
                    newArrayEntryMov[0,p,1] = np.nan
                    newArrayEntryVel[0,p,1] = np.nan
            movArrayOut = np.append(movArrayOut,newArrayEntryMov,axis=0)
            velArrayOut = np.append(velArrayOut,newArrayEntryVel,axis=0)

    # Modify movement and velocity arrays to be for movements in rather than movements out
    movArrayIn = np.copy(movArrayOut)
    velArrayIn = np.copy(velArrayOut)
    movArrayIn[:,:,1] = np.nan
    velArrayIn[:,:,1] = np.nan

    ind = np.where([True if np.isnan(movArrayOut[i,0,1]) else False for i in range(movArrayOut.shape[0])])[0]
    loVect = np.insert(ind[0:(-1)],0,0)
    hiVect = ind - 1
    for lo,hi in zip(loVect,hiVect):
        for k in range(lo,hi+1):
            movArrayIn[k+1,:,1] = movArrayOut[k,:,1]
            velArrayIn[k+1,:,1] = velArrayOut[k,:,1]



    # Next, create interpolated arrays by iterating over NGAs
    movArrayOutInterp = np.empty(shape=(0,9,2)) # Initialize the flow array
    flowInfoInterp = pd.DataFrame(columns=['NGA','Time']) # Initialize the info dataframe
    interpTimes = np.arange(-9600,1901,100)
    for nga in NGAs:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        for i_t,t in enumerate(interpTimes):
            if t >= min(times) and t <= max(times) and ((t in times) or (t in [time+100 for time in times])): # Is the time in the NGAs range?
                newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
                flowInfoInterp = flowInfoInterp.append(newInfoRow,ignore_index=True)
                newArrayEntry = np.empty(shape=(1,9,2))
                for p in range(movArrayOutInterp.shape[1]):
                    # Interpolate using flowArray
                    indFlow = flowInfo['NGA'] == nga
                    tForInterp = np.array(flowInfo['Time'][indFlow],dtype='float64')
                    pcForInterp = movArrayOut[indFlow,p,0]
                    currVal = np.interp(t,tForInterp,pcForInterp)
                    newArrayEntry[0,p,0] = currVal
                    if i_t < len(interpTimes) - 1:
                        nextTime = interpTimes[i_t + 1]
                        nextVal = np.interp(nextTime,tForInterp,pcForInterp)
                        newArrayEntry[0,p,1] = nextVal - currVal
                    else:
                        newArrayEntry[0,p,1] = np.nan
                movArrayOutInterp = np.append(movArrayOutInterp,newArrayEntry,axis=0)

    r0 = 1.5
    minPoints = 20
    dGrid = .2
    u0Vect,v0Vect = createGridForPC12(dGrid,velArrayOut)
    velScaling = 100
    return worldRegions,NGAs,PC_matrix,CC_df,CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling

#Read data generated in seshat.py
worldRegions,NGAs,PC_matrix,CC_df,CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()

tfd = tfp.distributions
sigma2_list = np.tile(np.geomspace(.25,2.5,num=10,endpoint=True).reshape([-1,1]),[1,9])
#sigma2 = np.power( ( (4/(dim+2))**(1/(dim+4)) )*(n**(-1/(dim+4))) *sig_xy, 2) Silverman's rule
u0Vect_out,v0Vect_out = createGridForPC12(dGrid,velArrayOut)
x_grids_out =  np.concatenate( (np.repeat(u0Vect_out,len(v0Vect_out)).reshape([-1,1]), np.tile( v0Vect_out, len(u0Vect_out) ).reshape([-1,1]) ) ,axis=1)
u0Vect_in,v0Vect_in = createGridForPC12(dGrid,velArrayIn)
x_grids_in =  np.concatenate( (np.repeat(u0Vect_in,len(v0Vect_in)).reshape([-1,1]), np.tile( v0Vect_in, len(u0Vect_in) ).reshape([-1,1]) ) ,axis=1)

flowInfo['NGA_id']=flowInfo.groupby('NGA').ngroup()
flowInfo['ID_within_NGA'] = flowInfo.groupby('NGA_id')['NGA_id'].rank(method='first')

##"OUT" data
d_x_out = velArrayOut[:,:2,0] #The starting point of "OUT" vector in the first 2 PC space
d_y_out = velArrayOut[:,2:,0] #The starting point of "OUT" vector in the other 7 PC space
d_v_out = velArrayOut[:,:2,1] #The "OUT" velocity in the first 2 PC space
d_w_out = velArrayOut[:,2:,1] #The "OUT" velocity in the other 7 PC space
d_xy_out = velArrayOut[:,:,0] #The starting point of "OUT" vector in 9 PC space

pos_v_not_nan_out = np.where(~np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of non-NaN points due to end point

n_obs_out = len(pos_v_not_nan_out)

#Removing NaN
d_x_notnan_out = d_x_out[pos_v_not_nan_out,:]
d_xy_notnan_out = d_xy_out[pos_v_not_nan_out,:]
d_v_notnan_out = d_v_out[pos_v_not_nan_out,:]

#fit GMM
gmm_y_fit_out = GMM(n_components=2).fit(d_y_out)
cov_out = gmm_y_fit_out.covariances_
mean_out = gmm_y_fit_out.means_
weights_out = gmm_y_fit_out.weights_

#sample

gmm_y_sample_out = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=weights_out),
    components_distribution=tfd.MultivariateNormalFullCovariance(
      loc=mean_out,       # One for each component.
      covariance_matrix=cov_out))  # And same here.

sig_xy_out = np.std(d_xy_out,axis=0)
dim = 9
n_out = d_y_out.shape[0]

##"IN" data
d_x_in = velArrayIn[:,:2,0] #The ending point of "IN" vector in the first 2 PC space
d_y_in = velArrayIn[:,2:,0] #The ending point of "OUT" vector in the other 7 PC space
d_v_in = velArrayIn[:,:2,1] #The "IN" velocity in the first 2 PC space
d_w_in = velArrayIn[:,2:,1] #The "IN" velocity in the other 7 PC space
d_xy_in = velArrayIn[:,:,0] #The ending point of "OUT" vector in 9 PC space

pos_v_not_nan_in = np.where(~np.isnan(d_v_in))[0][::2].astype(np.int32) #Position of non-NaN points due to starting point

n_obs_in = len(pos_v_not_nan_in)

d_xy_tf_in = tf.constant(d_xy_in[pos_v_not_nan_in,:],dtype=tf.float32)
d_v_tf_in = tf.constant(d_v_in[pos_v_not_nan_in,:],dtype=tf.float32) #Removed NaN already

d_x_notnan_in = d_x_in[pos_v_not_nan_in,:]
d_xy_notnan_in = d_xy_in[pos_v_not_nan_in,:]
d_v_notnan_in = d_v_in[pos_v_not_nan_in,:]

#fit GMM
gmm_y_fit_in = GMM(n_components=2).fit(d_y_in)
cov_in = gmm_y_fit_in.covariances_
mean_in = gmm_y_fit_in.means_
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
n_samples = 10000
sigma2_list = np.tile(np.geomspace(.25,2.5,num=10,endpoint=True).reshape([-1,1]),[1,9])

def Est_Flow_NN(d_x,d_v,x_to_pred=x_grids_out,v_to_pred=np.zeros([x_grids_out.shape[0],2]),wid1=20,wid2=20,layer=2,BATCH_SIZE='All',STEP_SIZE=20000,LEARNING_RATE=1e-3,dim=2, lag=1,l1_w = .1):

    tf.reset_default_graph()


    d_size = d_x.shape[0]
    v0 = d_v[:,0].reshape([-1,1])
    v1 = d_v[:,1].reshape([-1,1])

    # Define the input tensors and true output tensors
    X = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.int32, [None,1])

    # Initializer to set vectors to random initial values
    rnd_initializer = tf.initializers.random_normal(stddev=0.01)

    # Middle hidden layer with 100 neurons, W1 is weight matrix and b1 is bias vector
    wid1 = wid1
    wid2 = wid2

    W1 = tf.get_variable('W1', shape=[dim, wid1], initializer=rnd_initializer)
    b1 = tf.get_variable('b1', shape=[wid1]     , initializer=rnd_initializer)

    W2 = tf.get_variable('W2', shape=[wid1, wid2], initializer=rnd_initializer)
    b2 = tf.get_variable('b2', shape=[wid2]     , initializer=rnd_initializer)

    l1_norm = tf.get_variable('l1_norm', shape=[1], initializer=rnd_initializer)


    # Output layer, W2 is weight matrix and b2 is bias vector
    if layer is 1:
        W_o = tf.get_variable('W_o', shape=[wid1, 1], initializer=rnd_initializer)
        l1_norm = tf.reduce_sum(tf.abs(W1) )+tf.reduce_sum(tf.abs(b1) ) +tf.reduce_sum(tf.abs(W_o) ) +tf.reduce_sum(tf.abs(b_o))
    if layer is 2:
        W_o = tf.get_variable('W_o', shape=[wid2, 1], initializer=rnd_initializer)
        l1_norm = tf.reduce_sum(tf.abs(W1) )+tf.reduce_sum(tf.abs(b1) ) +tf.reduce_sum(tf.abs(W2) )+tf.reduce_sum(tf.abs(b2) ) +tf.reduce_sum(tf.abs(W_o) ) +tf.reduce_sum(tf.abs(b_o))


    b_o = tf.get_variable('b_o', shape=[1], initializer=rnd_initializer)

    middle_layer  = tf.nn.relu(tf.matmul(X, W1) + b1)

    middle_layer2 = tf.nn.relu(tf.matmul(middle_layer, W2) + b2)

    if layer is 1:
        pred_y        = tf.matmul(middle_layer, W_o) + b_o
    if layer is 2:
        pred_y        = tf.matmul(middle_layer2, W_o) + b_o

    mse = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred_y)) + l1_w*l1_norm

    BATCH_SIZE = BATCH_SIZE
    if BATCH_SIZE is 'All':
        BATCH_SIZE = d_size
    BATCHES_PER_DATASET = int(d_size/BATCH_SIZE)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse)

    vel_pred = np.zeros(2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(STEP_SIZE):
            if epoch%100 == 0:
#                 print("epoch %i:"%epoch)
                error = mse.eval(feed_dict={X:d_x,y:v0})
#                 print('error:'+str(error))

            for batch in range(BATCHES_PER_DATASET):
                c_batch_ixs = np.random.choice(d_size, BATCH_SIZE)
                sess.run(train_step, feed_dict={X: d_x[c_batch_ixs,:], y: v0[c_batch_ixs]})

            '''
            if dim is 2:
                vel_est0 = pred_y.eval(feed_dict={X:x_grids_out})
            if dim is 9:
                vel_est0 = pred_y.eval(feed_dict={X:np.concatenate((x_grids_out,np.zeros([x_grids_out.shape[0],7])),axis=1)  })
            '''


            if x_to_pred is not None:
                vel_pred0 = pred_y.eval(feed_dict={X:x_to_pred})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(STEP_SIZE):
            if epoch%100 == 0:
#                 print("epoch %i:"%epoch)
                error = mse.eval(feed_dict={X:d_x,y:v1})
#                 print('error:'+str(error))

            for batch in range(BATCHES_PER_DATASET):
                c_batch_ixs = np.random.choice(d_size, BATCH_SIZE)
                sess.run(train_step, feed_dict={X: d_x[c_batch_ixs,:], y: v1[c_batch_ixs]})

            '''
            if dim is 2:
                vel_est1 = pred_y.eval(feed_dict={X:x_grids_out})
            if dim is 9:
                vel_est1 = pred_y.eval(feed_dict={X:np.concatenate((x_grids_out,np.zeros([x_grids_out.shape[0],7])),axis=1)  })
            '''

            if x_to_pred is not None:
                vel_pred1 = pred_y.eval(feed_dict={X:x_to_pred})

    #v_est_grid = np.concatenate((vel_est0,vel_est1),axis=1)

    if x_to_pred is not None:
        vel_pred = np.concatenate((vel_pred0,vel_pred1),axis=1)
        pred_error = np.sum(np.sqrt(np.sum(np.power(v_to_pred-vel_pred,2),axis=1)))

    #return v_est_grid,pred_error,vel_pred
    return pred_error,vel_pred

# d_size = d_x_notnan_interp.shape[0]

# param_for_grid=[{"wid1":[20],"wid2":[20],"layer":[2],"BATCH_SIZE":['All'],'STEP_SIZE':[20000],'LEARNING_RATE':[1e-3],'dim':[2]}]

# param_list = list(ParameterGrid(param_for_grid))
# param_list = list(map(dict, set(tuple(sorted(d.items())) for d in param_list)) )
# # print(param_list)
d_x = d_x_notnan_interp
d_v = d_v_notnan_interp
# pred_error,vel_pred = Est_Flow_NN(d_x,d_v,x_to_pred=x_grids_out,v_to_pred=np.zeros([x_grids_out.shape[0],2]),**param_list[0])
# print(len(vel_pred) == len(d_x))
# print(len(vel_pred), len(d_x))
# pt = progress_timer(n_iter = 5000, description = 'velocity grid point estimation')

def bstr_flow(function, *args, n):
    """
    Given input arguments to the function, perform bootstrapping by resampling
    """
    vals = [] # the primary value of our interest on which we perform bootstrapping

    args = [np.asarray(i) for i in args]
    assert all(len(i) == len(args[0]) for i in args) # check all the inputs have the same length

    for i in range(n):

        resample = np.random.randint(0, len(args[0]), size=len(args[0]))
        resampled = [i[resample] for i in args]

        d_x_resampled, d_v_resampled = resampled[0], resampled[1]
        pred_error, vel_pred = function(d_x_resampled, d_v_resampled)
        vals.append(vel_pred)

	# mean_vel_preds = sum(vals)/n
	# vel_preds = [np.square(vel_pred-mean_vel_preds) for vel_pred in vals]
	# std = np.sqrt(sum(vel_preds)/(n-1))
    return vals

num_cores = mp.cpu_count()
num = int(5000/num_cores)
pool = Pool(processes=mp.cpu_count())
total = num*num_cores
results = [pool.apply_async(bstr_flow, (Est_Flow_NN, d_x, d_v, num)) for i in range(num_cores)]
vel_preds = [p.get() for p in results][0]
mean_preds=  sum(vel_preds)/total
vel_preds = [np.square(vel_pred-mean_preds) for vel_pred in vel_preds]
std = np.sqrt(sum(vel_preds)/(total-1))

with open('std1.txt', 'wb') as f:
	np.save(f, std, allow_pickle = False)

with open('mean1.txt', 'wb') as g:
	np.save(g, mean, allow_pickle = False)
