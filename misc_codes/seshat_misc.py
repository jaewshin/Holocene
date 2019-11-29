#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:55:40 2019

@author: hajime
"""


'''
Asymmetric regresion and MHG graphing.

'''

#%%
'''
Asymmetry regression
'''
from sklearn.linear_model import LinearRegression

CC_scaled_df = pd.DataFrame(CC_scaled,columns=[ 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money' ])
CC_scaled_df[['NGA','Time']] = CC_df[['NGA','Time']]


CC_reshape = CC_scaled_df.groupby(['NGA','Time']).mean().reset_index()
CC_fwd = CC_reshape.groupby(['NGA']).shift(-1)

CC_out = CC_fwd[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']] - CC_reshape[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']] 
CC_out['NGA'] = CC_reshape['NGA']

CC_out_vel = CC_out[['Time', 'PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']].div(CC_out['Time'],axis=0) *100.
CC_out_vel.columns = [str(col) + '_vel' for col in CC_out_vel.columns]
CC_out_vel['NGA'] = CC_out['NGA']

y = velArrayOut[:,0,1].reshape([-1,1])


##Average vel of CCs
CC_vel_mean = CC_out_vel.mean()

##Average vel of CCs when PC1 vel is positive
CC_vel_mean_PC1vel_positive = CC_out_vel.loc[(y>0).flatten() ].mean()

##Average vel of CCs when PC1 vel is positive
CC_vel_mean_PC1vel_negative = CC_out_vel.loc[(y<=0).flatten() ].mean()

#%%




#%%
##MGH plot

MHG_df = pd.read_csv('first_dates.csv') # A pandas dataframe
MHG_df = MHG_df.loc[MHG_df.MoralisingGods.notna()]

NGA_14 =['Big Island Hawaii',
 'Cuzco',
 'Deccan',
 'Kachi Plain',
 'Kansai',
 'Konya Plain',
 'Middle Yellow River Valley',
 'Niger Inland Delta',
 'North Colombia',
 'Orkhon Valley',
 'Paris Basin',
 'Sogdiana',
 'Susiana',
 'Upper Egypt']

plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
scale=100.
for i in range(0,velArrayOut.shape[0]):
    if not np.isnan(velArrayOut[i,0,1]):
        if flowInfo.NGA[i] in MHG_df.NGA.values:
            if flowInfo.NGA[i] in NGA_14:
                MHG_year = MHG_df.loc[MHG_df.NGA==flowInfo.NGA[i]].MoralisingGods.values[0]
                DM_year = MHG_df.loc[MHG_df.NGA==flowInfo.NGA[i]].DoctrinalMode.values[0]
                if flowInfo.Time[i]>=MHG_year:
                    rgb = cm.hsv(0)
                elif flowInfo.Time[i]>=DM_year:
                    rgb = cm.hsv(100)
                    
                else:
                    rgb = cm.hsv(50)            
                plt.arrow(velArrayOut[i,0,0],velArrayOut[i,1,0],velArrayOut[i,0,1]*scale,velArrayOut[i,1,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)
                
                if flowInfo.Time[i]==MHG_year:
                    plt.plot(velArrayOut[i,0,0],velArrayOut[i,1,0], 'bo')
            
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("OUT_in_the_data_MHG_DM_14NGAs.pdf")
plt.show()
plt.close()




#%%
#%%
#01/25 filter data points that i) have time gap and ii) move to different gaussian
#Fit GMM on 2D space
gmm_x_fit = GMM(n_components=2).fit(d_x_pos)
gmm_x_cov = gmm_x_fit.covariances_
gmm_x_mean = gmm_x_fit.means_
gmm_x_weights = gmm_x_fit.weights_



gmm_x_prob = gmm_x_fit.predict_proba(d_x_pos_notnan)
gmm_x_pred = gmm_x_fit.predict(d_x_pos_notnan)


d_x_pos_next_obs = d_x_pos_notnan + d_mov_notnan
gmm_x_prob_next = gmm_x_fit.predict_proba(d_x_pos_next_obs)
gmm_x_pred_next = gmm_x_fit.predict(d_x_pos_next_obs)

dt_notnan = dt[pos_v_not_nan_out]

filtered_loc = ~( (gmm_x_pred!=gmm_x_pred_next)*(dt_notnan.flatten()>100) )


d_x_pos_filtered = d_x_pos_notnan[filtered_loc]
d_x_dt_filtered = d_x_dt_notnan[filtered_loc]
d_v_filtered = d_v_notnan[filtered_loc]
d_mov_filtered = d_mov_notnan[filtered_loc]

d_x_pos_fil_normalized,d_x_pos_fil_mean,d_x_pos_fil_std = normalize(d_x_pos_filtered)
d_x_dt_fil_normalized,d_x_dt_fil_mean,d_x_dt_fil_std = normalize(d_x_dt_filtered)
d_v_fil_normalized,d_v_fil_mean,d_v_fil_std = normalize(d_v_filtered)
d_mov_fil_normalized,d_mov_fil_mean,d_mov_fil_std = normalize(d_mov_filtered)


x_grid_pos_fil_normalized = (x_grids_out-d_x_pos_fil_mean)/d_x_pos_fil_std
x_grid_dt_fil_normalized = (x_grid_dt-d_x_dt_fil_mean)/d_x_dt_fil_std


dict_for_matlab = {}
dict_for_matlab['velArrayOut_filtered'] = velArrayOut[pos_v_not_nan_out][filtered_loc]
dict_for_matlab['velArrayOut_notnan'] = velArrayOut[pos_v_not_nan_out]
dict_for_matlab['Gaussian'] = gmm_x_pred+1


#savemat('velArrayOut_plus_Gaussian.mat',dict_for_matlab)

'''
#Graph Original Flow with filtered data

plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
scale=100.

flowInfo_NGA_filtered = np.array(flowInfo.NGA[pos_v_not_nan_out][filtered_loc] )

for i in range(0,d_x_pos_filtered.shape[0]):
    rgb = cm.hsv(np.where(np.unique(flowInfo_NGA_filtered) == flowInfo_NGA_filtered[i])[0][0]/len(np.unique(flowInfo_NGA_filtered)))
    plt.arrow(d_x_pos_filtered[i,0],d_x_pos_filtered[i,1],d_v_filtered[i,0]*scale,d_v_filtered[i,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("OUT_in_the_data_filtered.pdf")
plt.show()
plt.close()


#Flows that are filtered out
#Graph Original Flow with filtered data
plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
scale=100.

flowInfo_NGA_removed = np.array(flowInfo.NGA[pos_v_not_nan_out][~filtered_loc] )

for i in range(0,d_x_pos_notnan[~filtered_loc].shape[0]):
    #rgb = cm.hsv(np.where(np.unique(flowInfo_NGA_filtered) == flowInfo_NGA_filtered[i])[0][0]/len(np.unique(flowInfo_NGA_filtered)))
    plt.arrow(d_x_pos_notnan[~filtered_loc][i,0],d_x_pos_notnan[~filtered_loc][i,1],d_v_notnan[~filtered_loc][i,0]*scale,d_v_notnan[~filtered_loc][i,1]*scale,width=.01,head_width=.06,head_length=.04,color=rgb)
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.savefig("removed_data_by_filtering.pdf")
plt.show()
plt.close()

'''

