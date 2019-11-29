# ###############################################################################
# # Generating Moralizing Gods Histogram (Supplementary Figure 13)
# ###############################################################################
# # The following coloring scheme is used for the Moralizing Gods plots:
# #
# # NA / Missing      Grey
# # Absent            Blue
# # Present           Green
# # Scatter Points    Red
mgMissCol = 'grey'
mgAbsnCol = 'blue'
mgPresCol = 'green'
mgScatCol = 'red'
#mgMissCol = [1,1,1] # grey
#mgAbsnCol = [0,0,1]    # blue
#mgPresCol = [0,1,0]    # green
#mgScatCol = [1,0,0]    # red
# Split data into missing, present, and absent
mgMiss = []
mgAbs = []
mgPres = []
for r in range(0,CC_df.shape[0]):
    nga = CC_df.loc[r]['NGA']
    tm = CC_df.loc[r]['Time']
    pc1Val = PC_matrix[r,0]
    # Handle some special cases due to how the two different data sets were generated
    if nga in ['Finger Lakes','Southern China Hills']:
      mgVal = np.nan
    elif nga in ['Garo Hills','Lena River Valley','Oro PNG']:
      mgVal = 0
    elif nga in ['Yemeni Coastal Plain']:
      mgVal = 1
    else:
        ind_mg = np.where((mg_df['NGA'].values == nga) & (mg_df['Time'].values == tm)) 
        if len(ind_mg) != 1:
            raise Exception('Bad matching for mg_df entry')
        ind_mg = ind_mg[0][0]
        mgVal = mg_df.loc[ind_mg]['MoralisingGods']
    if np.isnan(mgVal):
      mgMiss.append(pc1Val)
    elif mgVal == 0:
      mgAbs.append(pc1Val)
    elif mgVal == 1:
      mgPres.append(pc1Val)
    else:
        raise Exception('Unsupported MoralisingGods value')

# Make stacked histogram
plt.hist((mgMiss,mgAbs,mgPres),20,stacked=True,color=[mgMissCol,mgAbsnCol,mgPresCol]) 
plt.xlabel("Projection onto first Principal Component")
plt.ylabel("Counts")
plt.legend(['Missing','Absent','Present'])
fileStem = "pc1_histogram_mg"
plt.savefig(fileStem + ".pdf")
plt.close()
print("Done with pc1 histogram with mg (Supplementary Figure 13)")

##############################################################################
# Creating movement plot [in the spirit of OUT_in_the_data] (Figure 7)
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
        plt.arrow(movArrayOut[i,0,0],movArrayOut[i,1,0],movArrayOut[i,0,1],movArrayOut[i,1,1],width=lineWidth,head_width=headWidth,head_length=headLength,color=rgb)
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
plt.savefig("pc12_movement_plot.pdf")
plt.close()
print("Done with pc12 movmeent plot (Figure 7)")

###############################################################################
# Generating data for Moralizing Gods table 
###############################################################################
mg_df = pd.read_csv('./mhg_code/data_used_for_nature_analysis.csv')
nat_df = pd.read_csv('41586_2019_1043_MOESM6_ESM_sheet1.csv')
NGAs_nat = np.unique(nat_df['NGA'].values)
mg_tab = pd.DataFrame(columns=['NGA','Start','Stop','Value','Nature'])
for nga in NGAs:
    mg = mg_df.loc[mg_df['NGA'] == nga]['MoralisingGods'].values
    mg[np.isnan(mg)] = -1
    tm = mg_df.loc[mg_df['NGA'] == nga]['Time'].values
    seq = identifSeq(mg)
    for val,start,stop in seq:
        if val == -1:
            valStr = 'NA'
        elif val == 0:
            valStr = 'Absent'
        else:
            valStr = 'Present'

        if nga in NGAs_nat:
            natStr = 'Used'
        else:
            natStr = 'Ignored'
        mg_tab = mg_tab.append({'NGA':nga,'Start':tm[start],'Stop':tm[stop],'Value':valStr,'Nature':natStr},ignore_index=True)
mg_tab.to_latex('moralizing_gods_latex_table.txt')
print("Done with moralizing god table")

# ###############################################################################
# # Generating Moralizing Gods Histogram (Figure 8)
# ###############################################################################
# # The following coloring scheme is used for the Moralizing Gods plots:
# #
# # NA / Missing      Grey
# # Absent            Blue
# # Present           Green
# # Scatter Points    Red
mgMissCol = 'grey'
mgAbsnCol = 'blue'
mgPresCol = 'green'
mgScatCol = 'red'
#mgMissCol = [1,1,1] # grey
#mgAbsnCol = [0,0,1]    # blue
#mgPresCol = [0,1,0]    # green
#mgScatCol = [1,0,0]    # red
# Split data into missing, present, and absent
mgMiss = []
mgAbs = []
mgPres = []
for r in range(0,CC_df.shape[0]):
    nga = CC_df.loc[r]['NGA']
    tm = CC_df.loc[r]['Time']
    pc1Val = PC_matrix[r,0]
    # Handle some special cases due to how the two different data sets were generated
    if nga in ['Finger Lakes','Southern China Hills']:
      mgVal = np.nan
    elif nga in ['Garo Hills','Lena River Valley','Oro PNG']:
      mgVal = 0
    elif nga in ['Yemeni Coastal Plain']:
      mgVal = 1
    else:
        ind_mg = np.where((mg_df['NGA'].values == nga) & (mg_df['Time'].values == tm)) 
        if len(ind_mg) != 1:
            raise Exception('Bad matching for mg_df entry')
        ind_mg = ind_mg[0][0]
        mgVal = mg_df.loc[ind_mg]['MoralisingGods']
    if np.isnan(mgVal):
      mgMiss.append(pc1Val)
    elif mgVal == 0:
      mgAbs.append(pc1Val)
    elif mgVal == 1:
      mgPres.append(pc1Val)
    else:
        raise Exception('Unsupported MoralisingGods value')

# Make stacked histogram
plt.hist((mgMiss,mgAbs,mgPres),20,stacked=True,color=[mgMissCol,mgAbsnCol,mgPresCol]) 
plt.xlabel("Projection onto first Principal Component")
plt.ylabel("Counts")
plt.legend(['Missing','Absent','Present'])
fileStem = "pc1_histogram_mg"
plt.savefig(fileStem + ".pdf")
plt.close()
print("Done with pc1 histogram with mg (Figure 8)")

###############################################################################
# Creating movement plot for Moralizing Gods (Figure 9)
###############################################################################
plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
plt.xticks(size=25)
plt.yticks(size=25)

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
            
        
        plt.arrow(movArrayOut[i,0,0],movArrayOut[i,1,0],movArrayOut[i,0,1],movArrayOut[i,1,1],width=lineWidth,head_width=headWidth,head_length=headLength,color=rgb,alpha=.5,zorder=1)
        
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
plt.savefig("pc12_movement_plot_colored_by_MoralisingGods.pdf")
plt.close()
print("Done with pc12 movement plot with mg (Figure 9)")

# # %%
# ## Call the cross validation code
# #execfile('estimate_velocity.py')

###############################################################################
# Simulation models (Figure 10, 11)
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
        plt.savefig('S1.png')
    else:
        plt.savefig('S2.png')

    plt.clf()
    plt.close()

    print(f"Done with simulation models (Figure{figure})")

#Figure 10
simulation_plots(1, 1, -10, 5, 1, -1, 10)

#Figure 11
simulation_plots(1, 1, -10, 5, 1, -5, 11)

#########################################################################################
# Bootstrapping on the eigenvalues of each component and the angle between them (Figure 12)
#########################################################################################

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

plt.savefig("bootstrap.pdf")
plt.close()

###############################################################################
# Create a 5 x 2 plot to show time sequences organized by the ten world regions (Figure 13)
###############################################################################
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
plt.savefig("pc1_vs_time_stacked_by_region.pdf")
plt.close()

###############################################################################
# Interpolation plots (Figure 14)
###############################################################################

dataPath1 = os.path.abspath(os.path.join("./..","Seshat_arxiv","pnas_data1.csv")) #20 imputed sets
dataPath2 = os.path.abspath(os.path.join("./..","Seshat_arxiv","pnas_data2.csv")) #Turchin's PCs

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

fig.savefig('validation_interpolation.pdf')
plt.close()

print("Done with interpolation plots (Figure 14)")

###############################################################################
# Generating Moralizing Gods Stacked Plots (Figure 15)
###############################################################################
# Create a 5 x 2 plot to show time sequences organized by the ten world regions
print(worldRegions, type(worldRegions))
WorldRegions = dict()

WorldRegions['Africa'] = ['Upper Egypt', 'Niger Inland Delta', 'Ghanaian Coast']
WorldRegions['Europe'] = ['Latium', 'Paris Basin', 'Iceland']
WorldRegions['Central Eurasia'] = ['Susiana', 'Konya Plain', 'Yemini Coastal Plain']
WorldRegions['Southwest Asia'] = []
WorldRegions['South Asia'] = ['Cambodian basin']
WorldRegions['East Asia'] = ['Middle Yellow River Valley', 'Kansai', 'Southern China hills']
WorldRegions['North America'] = ['Valley of Oaxaca', 'Cahokia', 'Finger Lakes']
WorldRegions['South America'] = ['Cuzco', 'North Colombia', 'Lowland Andes']
WorldRegions['Oceania-Australia'] = ['Big Island Hawaii', 'Chuuk Islands', 'Oro PNG']
WorldRegions['Southeast Asia'] = []

f, axes = plt.subplots(nrows=int(len(NGAs)/2), ncols=2, sharex=True, sharey=True,figsize=(6,40))
for i,nga in enumerate(NGAs):
    indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
    times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
    pc1 = list()
    for t in times:
        ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
        pc1.append(np.mean(PC_matrix[ind,0]))
        
    mg = mg_df.loc[mg_df['NGA'] == nga]['MoralisingGods'].values
    mg[np.isnan(mg)] = -1
    tm = mg_df.loc[mg_df['NGA'] == nga]['Time'].values
    seq = identifSeq(mg)

    axes[i%int(len(NGAs)/2),i%2].set_facecolor('white')
    axes[i%int(len(NGAs)/2),i%2].spines['left'].set_color('black')
    axes[i%int(len(NGAs)/2),i%2].spines['right'].set_color('black')
    axes[i%int(len(NGAs)/2),i%2].spines['top'].set_color('black')
    axes[i%int(len(NGAs)/2),i%2].spines['bottom'].set_color('black')

    for val,start,stop in seq:
        if val == -1:
            rgb = mgMissCol
        elif val == 0:
            rgb = mgAbsnCol
        else:
            rgb = mgPresCol
        axes[i%int(len(NGAs)/2),i%2].add_patch(Rectangle((tm[start],pc1_min),tm[stop]-tm[start]+100,pc1_max-pc1_min,facecolor=rgb,edgecolor=rgb,zorder=1,alpha=.5))

    axes[i%int(len(NGAs)/2),i%2].scatter(times,pc1,s=10,color=mgScatCol,zorder=2)
    axes[i%int(len(NGAs)/2),i%2].set_xlim([t_min,t_max])
    axes[i%int(len(NGAs)/2),i%2].set_ylim([pc1_min,pc1_max])
    s = nga
    axes[i%int(len(NGAs)/2),i%2].set_title(s,fontsize=10)
    # if i % 3 == 2:
    #     plt.setp(axes[i%int(len(NGAs)/2),i%2].get_xticklabels(), visible=True)
    # else:
    #     plt.setp(axes[i%int(len(NGAs)/2),i%2].get_xticklabels(), visible=False)
    # if i == 29:
    #     axes[i%int(len(NGAs)/2),i%2].set_xlabel("Calendar Date [AD]")
    # axes[i%int(len(NGAs)/2),i%2].set_ylabel("PC1")

f.subplots_adjust(hspace=2)
plt.savefig("pc1_vs_time_stacked_with_mg_info.pdf")
#plt.show()
plt.close()
print("Done with pc1 vs mg info plot (Figure 15)")

#########################################################################################################
## Generate speed-adjusted histogram for PC1 and simulated histogram from Markov transition (taken out from the script)
#########################################################################################################
#
##%%
## Speed adjusted histogram (Figure 9)
#window_width=1.
#overlap=.5
#
#score_list = []
#center_list = []
#
#PC1_min = np.min(PC1)
#PC1_max = np.max(PC1)
#
#n_window = np.ceil( (PC1_max - PC1_min - window_width)/(window_width-overlap) ).astype(int)
#
#for i in range(n_window):
#    window = np.array([PC1_min+i*(window_width-overlap), PC1_min+i*(window_width-overlap)+window_width])
#    center = np.mean(window)
#    loc = (window[0]<=PC1) * (PC1<window[1])
#    
#    PC1_in_window = PC1[loc]
#    vel_PC1_in_window = vel_PC1[loc]
#    score = np.mean(1/np.abs(vel_PC1_in_window) )
#    center_list.append(center)
#    score_list.append(score)
#
#plt.axis()
#plt.xlim(-6,5)
##plt.ylim(-3,3)
#plt.plot(center_list, score_list)
#plt.xlabel("PC1 (center of window)")
#plt.ylabel("score")
## plt.title("speed-adjusted histogram(window %.3f, step %.3f)"%(window_width,window_width-overlap))
#plt.savefig("speed-adjusted-histogram.pdf")
##plt.show()
#plt.close()
