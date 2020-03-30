import time
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal, norm
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
# import seaborn as sns; sns.set()
from collections import defaultdict
import os
import math
import time
import pickle
from sklearn.decomposition import PCA, FastICA
from sklearn import linear_model
import scipy.spatial as spatial
from IPython.core.debugger import Pdb


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
    CC_file = "data1.csv" #20 imputed sets
    PC1_file = "data2.csv" #Turchin's PC1s
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
    velArrayOut = np.empty(shape=(0,9,3)) # Initialize the velocity array "Out" [location, movement / duration, duration]
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
            newArrayEntryVel = np.empty(shape=(1,9,3))
            for p in range(movArrayOut.shape[1]):
                newArrayEntryMov[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                newArrayEntryVel[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                if i_t < len(times) - 1:
                    nextTime = times[i_t + 1]
                    nextInd = indNga & (CC_df['Time']==nextTime) # boolean vector for slicing also by time
                    nextVal = np.mean(PC_matrix[nextInd,p])
                    newArrayEntryMov[0,p,1] = nextVal - newArrayEntryMov[0,p,0]
                    newArrayEntryVel[0,p,1] = newArrayEntryMov[0,p,1]/(nextTime-t)
                    newArrayEntryVel[0,p,2] = (nextTime-t)
                else:
                    newArrayEntryMov[0,p,1] = np.nan
                    newArrayEntryVel[0,p,1] = np.nan
                    newArrayEntryVel[0,p,2] = np.nan
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
            velArrayIn[k+1,:,2] = velArrayOut[k,:,2]



    # Next, create interpolated arrays by iterating over NGAs
    movArrayOutInterp = np.empty(shape=(0,9,2)) # Initialize the flow array 
    flowInfoInterp = pd.DataFrame(columns=['NGA','Time']) # Initialize the info dataframe
    interpTimes = np.arange(-9600,1901,100)
    for nga in NGAs:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        for i_t,t in enumerate(interpTimes):
            if t >= min(times) and t <= max(times): # Is the time in the NGAs range?
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

def identifSeq(v):
    # v is a vector-like object
    # identify sequences along with their ranges

    if(len(v) == 1):
      return([(v,0,0)])

    currentVal = v[0]
    currentIndex = 0
    seq = []
    for i,x in enumerate(v[1:]):
        if x != currentVal:
            seq.append((currentVal,currentIndex,i))
            currentVal = x
            currentIndex = i+1
    # Add the last entry
    seq.append((currentVal,currentIndex,i+1))
    return(seq)
