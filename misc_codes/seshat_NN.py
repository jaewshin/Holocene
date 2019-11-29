#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:39:00 2019

@author: hajime
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


#%%
#Estimate the flow field from simple neural network
##Input: 
##d_x: the position of observed flow
##d_v: observed flow
##x_to_pred: Predict the flow at this point, if necessary. Use for validation or estimate at grid point.
##v_to_pred: Compare this value to the estimate at x_to_pred for validation, if necessary.
##wid1, wid2: number of nodes in the first and second layer
##layer: number of hidden layer, 1 or 2.

def Est_Flow_NN(d_x,d_v,x_to_pred=None,v_to_pred=None,wid1=100,wid2=20,layer=2,BATCH_SIZE='All',STEP_SIZE=20000,LEARNING_RATE=1e-3,dim=3,lag=1,l1_w = .2,l2_w=0.):

    tf.reset_default_graph()
    
    
    d_size = d_x.shape[0]
    v0 = d_v[:,0].reshape([-1,1])
    v1 = d_v[:,1].reshape([-1,1])
    
    # Define the input tensors and true output tensors
    X = tf.placeholder(tf.float32, [None, dim])
    y = tf.placeholder(tf.float32, [None,1])

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
    l2_norm = tf.get_variable('l2_norm', shape=[1], initializer=rnd_initializer)


    # Output layer, W2 is weight matrix and b2 is bias vector
    if layer is 1:
        W_o = tf.get_variable('W_o', shape=[wid1, 1], initializer=rnd_initializer)
    if layer is 2:
        W_o = tf.get_variable('W_o', shape=[wid2, 1], initializer=rnd_initializer)
        
    b_o = tf.get_variable('b_o', shape=[1]     , initializer=rnd_initializer)

    middle_layer  = tf.nn.relu(tf.matmul(X, W1) + b1)
    
    middle_layer2 = tf.nn.relu(tf.matmul(middle_layer, W2) + b2)
    
    if layer is 1:
        pred_y        = tf.matmul(middle_layer, W_o) + b_o
        l1_norm = tf.reduce_sum(tf.abs(W1) )+tf.reduce_sum(tf.abs(b1) ) +tf.reduce_sum(tf.abs(W_o) ) +tf.reduce_sum(tf.abs(b_o) )
        l2_norm = tf.sqrt( tf.reduce_sum(tf.square(W1) )+tf.reduce_sum(tf.square(b1) ) +tf.reduce_sum(tf.square(W_o) ) +tf.reduce_sum(tf.square(b_o) ) )
    if layer is 2:
        pred_y        = tf.matmul(middle_layer2, W_o) + b_o
        l1_norm = tf.reduce_sum(tf.abs(W1) )+tf.reduce_sum(tf.abs(b1) ) +tf.reduce_sum(tf.abs(W2) )+tf.reduce_sum(tf.abs(b2) ) +tf.reduce_sum(tf.abs(W_o) ) +tf.reduce_sum(tf.abs(b_o) )
        l2_norm = tf.sqrt( tf.reduce_sum(tf.square(W1) )+tf.reduce_sum(tf.square(b1) ) +tf.reduce_sum(tf.square(W2) )+tf.reduce_sum(tf.square(b2) ) +tf.reduce_sum(tf.square(W_o) ) +tf.reduce_sum(tf.square(b_o) ) )

        
    mse = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=pred_y)) + l1_w*l1_norm + l2_w*l2_norm

    BATCH_SIZE = BATCH_SIZE
    if BATCH_SIZE == 'All':
        BATCH_SIZE = d_size
        BATCHES_PER_DATASET = 1
    else:
        BATCHES_PER_DATASET = int(d_size/BATCH_SIZE)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(mse)
    
    vel_pred = np.zeros(2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(STEP_SIZE):
            if epoch%5000 == 0: 
                print("epoch %i:"%epoch)
                error = mse.eval(feed_dict={X:d_x,y:v0})
                print('error:'+str(error))
                
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
            '''
            if epoch%1000 == 0: 
                print("epoch %i:"%epoch)
                error = mse.eval(feed_dict={X:d_x,y:v1})
                print('error:'+str(error))
            '''
                
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
    
    pred_error = None
    if x_to_pred is not None:    
        vel_pred = np.concatenate((vel_pred0,vel_pred1),axis=1)  
        if v_to_pred is not None:
            pred_error = np.sum(np.sqrt(np.sum(np.power(v_to_pred-vel_pred,2),axis=1)))
        
    #return v_est_grid,pred_error,vel_pred
    return pred_error,vel_pred



#k-fold cross validation
def Est_Flow_NN_CV(d_x,d_v,param,n_splits=5):
    kf = KFold(n_splits=n_splits)
    errors = []   
    j=0
    for train_index, test_index in kf.split(d_x):
        
        print('**split'+str(j)+'**')
        X_train, X_test = d_x[train_index], d_x[test_index]
        y_train, y_test = d_v[train_index], d_v[test_index]

        #vel_est,pred_error,_ = Est_Flow_NN(X_train,y_train,X_test,y_test,**param)
        pred_error,_ = Est_Flow_NN(X_train,y_train,X_test,y_test,**param)
        errors.append(pred_error)
        j = j+1
    error_mean = np.mean(errors)
    return error_mean





def bstr_flow(function,x_train,y_train,x_grid,function_param,n):
    """
    Given input arguments to the function, perform bootstrapping by resampling 
    """  
    vals = [] # the primary value of our interest on which we perform bootstrapping 
         
    #args = [np.asarray(i) for i in args]
    #assert all(len(i) == len(args[0]) for i in args) # check all the inputs have the same length

    T = x_train.shape[0]
    for i in range(n):
        
        resample = np.random.randint(0, T, size=T)
        x_train_i = x_train[resample]
        y_train_i = y_train[resample]
        
        #pred_error, vel_pred = function(d_x_resampled, d_v_resampled,args[2])
        _,x_change_pred = function(x_train_i,y_train_i,x_grid,**function_param)
        vals.append(x_change_pred)

	# mean_vel_preds = sum(vals)/n
	# vel_preds = [np.square(vel_pred-mean_vel_preds) for vel_pred in vals]
	# std = np.sqrt(sum(vel_preds)/(n-1))
    return vals



#%%
#01/28 NN with filtered dataset, finer L1,L2
'''

## with time gap
#param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[.0,.1,.2]}]
#param_for_grid=[{"wid1":[50],"wid2":[50],"layer":[2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[.0,.0125,.025,.05,.1]}]
#param_for_grid=[{"wid1":[50],"wid2":[50],"layer":[2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[0.,0.0001    , 0.00014384, 0.00020691, 0.00029764, 0.00042813,
#       0.00061585, 0.00088587, 0.00127427, 0.00183298, 0.00263665,
#       0.00379269, 0.00545559, 0.0078476 , 0.01128838, 0.01623777,
#       0.02335721, 0.03359818, 0.0483293 , 0.06951928, 0.1       ]
#}]
param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[.0],'l2_w':[.001,.01,.1]}]
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
    error[i] = Est_Flow_NN_CV(d_x=d_x_dt_fil_normalized,d_v=d_mov_fil_normalized,param=param_list[i])    
    _, pred = Est_Flow_NN(d_x=d_x_dt_fil_normalized,d_v=d_mov_fil_normalized,x_to_pred=x_grid_dt_fil_normalized ,**param_list[i]) 
    
    pred = pred*d_mov_fil_std+d_mov_fil_mean
    
    pred_grid.append(pred)
    
with open("CV_result_NN_dt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(error, f)
with open("CV_param_NN_dt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(param_list, f)
with open("CV_pred_NN_dt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(pred_grid, f)

#CV without time gap

#param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[2],'lag':[1],'l1_w':[.0,.1,.2]}]
#param_for_grid=[{"wid1":[50],"wid2":[50],"layer":[2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[2],'lag':[1],'l1_w':[.0,.0125,.025,.05,.1]}]
#param_for_grid=[{"wid1":[50],"wid2":[50],"layer":[2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[0.,0.0001    , 0.00014384, 0.00020691, 0.00029764, 0.00042813,
#       0.00061585, 0.00088587, 0.00127427, 0.00183298, 0.00263665,
#       0.00379269, 0.00545559, 0.0078476 , 0.01128838, 0.01623777,
#       0.02335721, 0.03359818, 0.0483293 , 0.06951928, 0.1       ]
#}]
param_for_grid=[{"wid1":[20,50,100],"wid2":[20,50,100],"layer":[1,2],"BATCH_SIZE":['All'],'STEP_SIZE':[40000],'LEARNING_RATE':[1e-3],'dim':[3],'lag':[1],'l1_w':[.0],'l2_w':[.001,.01,.1]}]

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
    
    start_time = time.time()

    
    error[i] = Est_Flow_NN_CV(d_x=d_x_pos_fil_normalized,d_v=d_v_fil_normalized,param=param_list[i])  
    _, pred = Est_Flow_NN(d_x=d_x_pos_fil_normalized,d_v=d_v_fil_normalized,x_to_pred=x_grid_pos_fil_normalized ,**param_list[i])  
    pred_grid.append(pred)
    
    pred = pred*d_v_fil_std+d_v_fil_mean


    end_time = time.time()
    time_elapsed = end_time-start_time
    print('time for a setting: ',time_elapsed)

    
with open("CV_result_NN_nodt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(error, f)
with open("CV_pred_NN_nodt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(pred_grid, f)
with open("CV_param_NN_nodt_filtered_190129_L2.pickle","wb") as f:
    pickle.dump(param_list, f)



'''




#%%
#Read result
'''
with open("CV_result_NN_dt_filtered_190128.pickle","rb") as f:
    cv_result_dt = pickle.load(f)
with open("CV_param_NN_dt_filtered_190128.pickle","rb") as f:
    cv_param_dt = pickle.load( f)
with open("CV_pred_NN_dt_filtered_190128.pickle","rb") as f:
    cv_pred_dt = pickle.load( f)

cv_pred_dt = cv_pred_dt * d_mov_std +d_mov_mean
    
best_param_dt = cv_param_dt[np.argmin(cv_result_dt)]

pred_best_dt = cv_pred_dt[np.argmin(cv_result_dt)]



with open("CV_result_NN_nodt_filtered_190128.pickle","rb") as f:
    cv_result_nodt = pickle.load(f)
with open("CV_param_NN_nodt_filtered_190128.pickle","rb") as f:
    cv_param_nodt = pickle.load( f)
with open("CV_pred_NN_nodt_filtered_190128.pickle","rb") as f:
    cv_pred_nodt = pickle.load( f)
    
cv_pred_nodt = cv_pred_nodt * d_v_std+ d_v_mean
    
best_param_nodt = cv_param_nodt[np.argmin(cv_result_nodt)]

pred_best_nodt = cv_pred_nodt[np.argmin(cv_result_nodt)]
'''




#%%
#01/16/2019 graph from NN with and without dt
'''





scale=50.
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)

for i in range(0,pred_best_nodt.shape[0]):
    plt.arrow(x_grid_dt[i,0],x_grid_dt[i,1],pred_best_nodt[i,0]*scale,pred_best_nodt[i,1]*scale,width=.01,head_width=.06,head_length=.04)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NN without time gap, longer iter (CVed)")
plt.savefig("NN without time gap, longer iter (CVed).pdf")
plt.show()
plt.close()






scale=1.
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)

for i in range(0,pred_best_dt.shape[0]):
    plt.arrow(x_grid_dt[i,0],x_grid_dt[i,1],pred_best_dt[i,0]*scale,pred_best_dt[i,1]*scale,width=.01,head_width=.06,head_length=.04)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NN with time gap, longer iter (CVed)")
plt.savefig("NN with time gap, longer iter (CVed).pdf")
plt.show()
plt.close()
'''

#%%
#Pck up one setting
'''
i_setting = 10
cv_pred_graph = cv_pred_nodt[i_setting]
scale=50.
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)

for i in range(0,pred_best_nodt.shape[0]):
    plt.arrow(x_grid_dt[i,0],x_grid_dt[i,1],cv_pred_graph[i,0]*scale,cv_pred_graph[i,1]*scale,width=.01,head_width=.06,head_length=.04)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("NN without time gap, longer iter")
plt.savefig("NN without time gap, longer iter.pdf")
plt.show()
plt.close()
'''

#%%
#Graph all the settings

'''
for i_setting in range(len(cv_param_nodt) ):

    cv_pred_graph = cv_pred_nodt[i_setting]
    scale=100.
    plt.axis('scaled')
    plt.xlim(-6,5)
    plt.ylim(-3,3)
    
    for i in range(0,pred_best_nodt.shape[0]):
        plt.arrow(x_grid_dt[i,0],x_grid_dt[i,1],cv_pred_graph[i,0]*scale,cv_pred_graph[i,1]*scale,width=.01,head_width=.06,head_length=.04)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("NN without time gap, L1norm: %f"%cv_param_nodt[i_setting]['l1_w'])
    #plt.savefig("NN_without_timegap_L1norm%f.pdf"%cv_param_nodt[i_setting]['l1_w'])
    plt.show()
    plt.close()

'''
