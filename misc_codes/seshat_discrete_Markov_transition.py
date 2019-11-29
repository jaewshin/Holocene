#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:47:54 2019

@author: hajime
"""

#%%
window_width=1.
overlap=.5

def hist_normalized_by_vel(PC1,vel_PC1,window_width=.5,overlap=.25):
    score_list = []
    center_list = []
    
    PC1_min = np.min(PC1)
    PC1_max = np.max(PC1)
    
    n_window = np.ceil( (PC1_max - PC1_min - window_width)/(window_width-overlap) ).astype(int)
    
    for i in range(n_window):
        window = np.array([PC1_min+i*(window_width-overlap), PC1_min+i*(window_width-overlap)+window_width])
        center = np.mean(window)
        loc = (window[0]<=PC1) * (PC1<window[1])
        
        PC1_in_window = PC1[loc]
        vel_PC1_in_window = vel_PC1[loc]
        score = np.mean(1/np.abs(vel_PC1_in_window) )
        center_list.append(center)
        score_list.append(score)
    
    plt.axis()
    plt.xlim(-6,5)
    #plt.ylim(-3,3)
    plt.plot(center_list, score_list)
    plt.xlabel("PC1 (center of window)")
    plt.ylabel("score")
    plt.title("speed-adjusted histogram(window %.3f, step %.3f)"%(window_width,window_width-overlap))
    plt.savefig("speed-adjusted histogram(window %.3f, step %.3f).pdf"%(window_width,window_width-overlap))
    plt.show()
    plt.close()
    
    return
def calc_transition_matrix(PC1,vel_PC1,n_bin=6):
    left_end = np.min(PC1+vel_PC1)
    right_end = np.max(PC1+vel_PC1)
    width = (right_end-left_end)/n_bin
    
    
    transition_count_matrix = np.zeros([n_bin,n_bin])
    transition_prob_matrix = np.zeros([n_bin,n_bin])
    for i in range(n_bin):
        loc_origin = (left_end+i*width <=PC1)*(PC1<left_end+(i+1)*width)
        num_origin_i = np.sum(loc_origin)
        if num_origin_i == 0:
            print('No observation starting in bin %i'%i)
        center = left_end+(i+.5)*width
        dest = center+vel_PC1[loc_origin]
        for j in range(n_bin):
            loc_dest_j = (left_end+j*width <=dest)*(dest<left_end+(j+1)*width)
            transition_count_matrix[i,j] = np.sum(loc_dest_j)
            transition_prob_matrix[i,j] = np.sum(loc_dest_j)/num_origin_i
    return transition_count_matrix,transition_prob_matrix

init_PC1 = d_x_out[pos_v_nan_in,0]
final_PC1 = d_x_out[pos_v_nan_out,0]
PC1 = d_x_notnan_out[:,0]
vel_PC1 = d_v_notnan_out[:,0]

def cum_transition(PC1,vel_PC1_annual,init_PC1, years_move=100,n_bin=6,n_iter=10 ,flag_barrier=False, graph=True,graph_name='' ,flag_rm_jump=False, transition_prob_input=None,ratio_init_input=None):
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
        
    print(ratio_init)
    
    
    #--------
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
    if graph:
        f, axarr = plt.subplots(n_iter,2,sharex='col',figsize=[7.5,25.])#,sharey='col'
        f.set_figheight(50)
        
    for i in range(n_iter):
        dist_transition.append(dist_i)
        dist_cum_transition.append(dist_cum_i)
        dist_i = np.matmul(transition_prob_matrix.T,dist_i)
        dist_cum_i = dist_cum_i + dist_i
        
        if graph:
            axarr[i,0].bar(center_list, dist_i.flatten() , color='b')
            axarr[i,0].set_title('Current dist at iter %i'%i)
            axarr[i,1].bar(center_list, dist_cum_i.flatten(),color='r' )
            axarr[i,1].set_title('Cumulative dist after iter %i'%i)
    #f.suptitle("simulated histogram from transition matrix (bins: %i, time scale: %i years)"%(n_bin,years_move))
    if graph:      
        f.subplots_adjust(hspace=.2)
        
        #plt.savefig("simulated histogram from transition matrix (bins %i, time scale %i years, barrier %i, %s).pdf"%(n_bin,years_move,flag_barrier,graph_name))
        plt.show()
        plt.close()
    
    return transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition
        
    
#years_move=500
#n_bin=15
#n_iter=20


years_move=500
n_bin=10
n_iter=20

flag_barrier=True    
flag_rm_jump = False
transition_count_matrix_b, transition_prob_matrix_b,ratio_init,_,_ = cum_transition(PC1,vel_PC1,init_PC1, years_move=years_move,n_bin=n_bin,n_iter=n_iter,flag_barrier=flag_barrier,flag_rm_jump=flag_rm_jump )    
#transition_count_matrix, transition_prob_matrix,ratio_init = cum_transition(PC1,vel_PC1,init_PC1, years_move=500,n_bin=10,n_iter=15,flag_barrier=False  )    

#%%
'''
NGA-specific iteration length
'''
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
print(transition_prob_matrix)
    

dist_sum_NGA = np.zeros(n_bin)
for i in range(30):
    dist_cum_final = np.sum(dist_transition_list[i],axis=0)
    dist_sum_NGA = (dist_sum_NGA +dist_cum_final.flatten() )
plt.bar(center_list, dist_sum_NGA, color='g')
plt.show()
#plt.savefig("simulated histogram from transition matrix, NGA-specific iteration length (bins %i, time scale %i years, barrier %i).pdf"%(n_bin,years_move,flag_barrier))
plt.close()

#%%
'''
Starting from uniform as distribution, NGA-specific iteration length
'''

dist_transition_list = []
dist_cum_transition_list = []
ratio_init_input = np.ones(n_bin)/n_bin
n_iter=10
for i in range(30):
    init = init_PC1[i] #doesn't matter
    
    transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter,flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False,ratio_init_input=ratio_init_input )    
    dist_transition_list.append(dist_transition)
    dist_cum_transition_list.append(dist_cum_transition)
print(transition_prob_matrix)
    

dist_sum_NGA = np.zeros(n_bin)
for i in range(30):
    dist_cum_final = np.sum(dist_transition_list[i],axis=0)
    dist_sum_NGA = (dist_sum_NGA +dist_cum_final.flatten() )
plt.bar(center_list, dist_sum_NGA, color='g')
plt.show()
plt.savefig("simulated histogram fstarting from uniform dist (iter %i, bins %i, time scale %i years, barrier %i).pdf"%(n_iter, n_bin,years_move,flag_barrier))
plt.close()



plt.bar(center_list, dist_sum_NGA, color='g')
plt.savefig("simulated histogram fstarting from uniform dist (iter %i, bins %i, time scale %i years, barrier %i).pdf"%(n_iter, n_bin,years_move,flag_barrier))
plt.show()
plt.close()


plt.bar(center_list[:n_bin-1], dist_sum_NGA[:n_bin-1], color='y')
plt.savefig("simulated histogram fstarting from uniform dist (no last bin) (iter %i, bins %i, time scale %i years, barrier %i).pdf"%(n_iter, n_bin,years_move,flag_barrier))
plt.show()
plt.close()

#%%
plt.imshow(transition_prob_matrix, cmap='hot')
plt.xticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.yticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.colorbar()
plt.savefig("heatmap of transition matrix (bins %i, time scale %i years, barrier %i).pdf"%(n_bin,years_move,flag_barrier))
plt.show()



#%%
a=np.arange(n_bin).reshape([1,-1] ) 
aa = a
for i in range(1,n_bin):
    aa=np.concatenate( (aa,a-i),axis=0 )
    
mean_step = np.sum(transition_prob_matrix*aa , axis=1)
var_step = np.sum(transition_prob_matrix*np.power(aa,2) , axis=1) - np.power(mean_step,2)
#%%
transition_count_matrix_temp = transition_count_matrix
transition_count_matrix_temp[:5,7:]=0
num_origin_temp=np.sum(transition_count_matrix_temp,axis=1)

transition_prop_matrix_temp = transition_count_matrix_temp/num_origin_temp.repeat(10).reshape([10,10])

plt.imshow(transition_prop_matrix_temp, cmap='hot')
plt.xticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.yticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.colorbar()
plt.show()

mean_step_temp = np.sum(transition_prop_matrix_temp*aa , axis=1)
var_step_temp = np.sum(transition_prop_matrix_temp*np.power(aa,2) , axis=1) - np.power(mean_step_temp,2)



#%%
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
    
#%%
q = transition_prob_matrix_b[1,:]
p_init = np.copy(q)

x=aa[1,:]
mu = mean_step[1]
sig2=var_step[6]
pos_to_empty = [7,8,9]

args = (q,x,mu,sig2,pos_to_empty)

bnds=((0.,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,0.),(0,1.),(0,1.),(0,1.))
cons1=({'type':'eq','fun':lambda p: np.abs( np.sum(p)-1. )  },{'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }, {'type':'eq','fun':lambda p: np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) }   )

sol1 = minimize(func_obj, p_init, args, method = 'SLSQP',options={'ftol':10e-10,'maxiter':5000},bounds=bnds,constraints=cons1) #
p1 =sol1.x

print(p1)
print( np.sum(p1) )
print( np.abs(np.sum(p1*x)-mu ) )
print( np.abs( np.sum(p1* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
print(  np.abs(p1[ np.where( q==0 ) ] ) )



#%%
q = transition_prob_matrix_b[2,:]
p_init = np.copy(q)

x=aa[2,:]
mu = mean_step[2]
sig2=var_step[6]

args = (q,x,mu,sig2,pos_to_empty)
#bnds=((0.,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.),(0,1.))
bnds=np.array([[0.,1.],[0,1.],[0,1.],[0,1.],[0,1.],[0,1.],[0,1.],[0,1.],[0,1.],[0,1.]])
cons2=({'type':'eq','fun':lambda p: np.abs( np.sum(p)-1. )  },{'type':'eq','fun':lambda p: np.abs(np.sum(p*x)-mu ) }, {'type':'eq','fun':lambda p: np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) }   )

sol2 = minimize(func_obj, p_init, args, method = 'SLSQP',options={'ftol':10e-10,'maxiter':5000},bounds=bnds,constraints=cons2)
p2 =sol2.x

print(p2)
print( np.sum(p2) )
print( np.abs(np.sum(p2*x)-mu ) )
print( np.abs( np.sum(p2* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
print(  np.abs(p2[ np.where( q==0 ) ] ) )







#%%
'''
Emptying the "jump" probability from 2nd and 3rd bin to the last 3 bins
'''
transition_prob_matrix_modified = np.copy(transition_prob_matrix)
transition_prob_matrix_modified[1,:] = p1
transition_prob_matrix_modified[2,:] = p2

def sim_trans(transition_prob_matrix,figtitle=None):

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
    plt.bar(center_list, dist_sum_NGA, color='g')
    plt.title(figtitle)
    plt.savefig("Match var  on 2nd-7th bins(bins %i, time scale %i years, barrier %i).pdf"%(n_bin,years_move,flag_barrier))
    plt.show()
    plt.close()


#%%
'''
Match the mean and/or variance of the transition starting from each bin (2nd to 8th).
Minimize KL divergence
'''
mean_all = np.sum(transition_count_matrix*aa )/np.sum(transition_count_matrix)
sig2_all = np.sum(transition_count_matrix*np.power(aa,2) )/np.sum(transition_count_matrix) - np.power(mean_all,2)

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
    print('---%i---'%i)
    print('prob:',p)
    print( 'sum of prob:',np.sum(p) )
    print( 'error in mean:',np.abs(np.sum(p*x)-mu ) )
    print( 'error in var:',np.abs( np.sum(p* np.power(x,2) ) -np.power(mu,2) -sig2 ) )
    print(  'pos q 0:',np.abs(p[ np.where( q==0 ) ] ) )
    
    transition_prob_matrix_modified[i,:] = p
#%%
sim_trans(transition_prob_matrix_modified,figtitle)    
