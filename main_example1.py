# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:17:24 2020

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:53:42 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
import pdb
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neighbors import KernelDensity
import time

#from scipy.linalg import cholesky, cho_solve, solve_triangular
#import numpy.linalg as LA
#import math



def gamma(x):
    return (x-5)**2 + 1.0
def evaluate(x, tag):
    sigma2 = 10.
    beta1 = 0.6
    beta2 = 0.3
    kpa = 0.2
    D = np.array([[-1,1],[1,-1]])
    # D = np.array([[0.8,-0.3],[-0.2,0.5]])
    gamma0 = gamma(x)[0]
    gamma1 = gamma(x)[1]
    if tag == 'g':
        minus_gradV0 = 0.5*sigma2/gamma0 - beta1*x[0] + kpa*(D[0,0]*x[0] + D[0,1]*x[1]) + 0.1*x[1]
        minus_gradV1 = 0.5*sigma2/gamma1 - beta2*x[1] + kpa*(D[1,0]*x[0] + D[1,1]*x[1])
        return np.array([minus_gradV0,minus_gradV1])
    elif tag == 'h':
        minus_H = np.zeros([2,2])
        minus_H[0,0] = -sigma2*(x[0] -5.)/gamma0**2 - beta1 + kpa*D[0,0]
#        pdb.set_trace()
        minus_H[0,1] = kpa*D[0,1] + 0.1*D[0,1]
        minus_H[1,0] = kpa*D[1,0]
        minus_H[1,1] = -sigma2*(x[1] - 5.)/gamma1**2 - beta2 + kpa*D[1,1]
        return minus_H
#def MLE_fun(params, x, obs):
#    Dim = obs.shape[0]
#    sigma2 = np.exp(params)[-1]
#    Y = obs[:,np.newaxis]
#    K =  comp_K(x,x,np.exp(params), "f_f")+sigma2*np.eye(Dim)
#    inv_K     = np.linalg.inv(K)
#    # pdb.set_trace()
#    log_likelihood_dims = -0.5 * np.mat(Y.T)*np.mat(inv_K)*np.mat(Y)
#    log_likelihood_dims -= np.log(np.diag(K)).sum()
#    log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
#    log_likelihood = log_likelihood_dims.sum(-1)
#    print('Log',log_likelihood )
#    return -(log_likelihood )  
def comp_K(X, Y, params, tag):
  alpha, length_scale, sigma2 = params
  # X and Y must be a 2-dimensional array
  RBF_yy = RBF(length_scale, (1e-2, 1e3))
  # pdb.set_trace()
  K_ff = RBF_yy(X, Y)
  dist0 = (np.tile(X[:,0:1].T,(Y.shape[0],1)) - np.tile(Y[:,0:1],(1,X.shape[0]))).T
  dist1 = (np.tile(X[:,1:].T,(Y.shape[0],1)) - np.tile(Y[:,1:],(1,X.shape[0]))).T
  if tag == "y_y":
    K = alpha * K_ff + sigma2*np.eye(K_ff.shape[0])
  elif tag == "y_f":
    K = alpha* K_ff
  elif tag == "y_g1":
    K = (dist0)/length_scale**2* alpha * K_ff
  elif tag == "y_g2":
    K = (dist1)/length_scale**2* alpha * K_ff
  elif tag == "f_y":
    K = alpha* K_ff
  elif tag == "f_f":
    K = alpha* K_ff 
  elif tag == "f_g1":
    K = (dist0)/length_scale**2* alpha * K_ff
  elif tag == "f_g2":
    K = (dist1)/length_scale**2* alpha * K_ff
  elif tag == "g1_f":
    K = -(dist0)/length_scale**2* alpha * K_ff
  elif tag == "g2_f":
    K = -(dist1)/length_scale**2* alpha * K_ff
  elif tag == "g1_g1":
    K = (-dist0**2/length_scale**4 + 1/(length_scale**2))*alpha*K_ff 
  elif tag == "g2_g2":
    K = (-dist1**2/length_scale**4 + 1/(length_scale**2))*alpha*K_ff 
  else :
    print("error!")
  return K
def predict_derivative(x_pre, X_data, obs, params,inv_KDD,tag):
  # Data_yAndr include y and r values of the  trainning data
    Data_yAndr = obs
    if tag =="f":
      # pdb.set_trace()
      Cov_ff = comp_K(x_pre, x_pre,params,"f_f")
      Cov_fY = comp_K(x_pre,X_data,params,"f_y")
      Cov_fD = Cov_fY
      mean = 0. + np.dot(np.dot(Cov_fD,inv_KDD),Data_yAndr-0.)
      Cov  = Cov_ff - np.dot(np.dot(Cov_fD,inv_KDD),np.transpose(Cov_fD))

    elif tag=="g1":
      Cov_gY = comp_K(x_pre,X_data,params,"g1_f")
      Cov_gD = Cov_gY
      Cov_gg = comp_K(x_pre, x_pre,params,"g1_g1")

      mean = 0. + np.dot(np.dot(Cov_gD,inv_KDD),Data_yAndr - 0.)
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_KDD),np.transpose(Cov_gD))
    elif tag=="g2":
      Cov_gY = comp_K(x_pre,X_data,params,"g2_f")
      Cov_gD = Cov_gY
      Cov_gg = comp_K(x_pre, x_pre,params,"g2_g2")

      mean = 0. + np.dot(np.dot(Cov_gD,inv_KDD),Data_yAndr-0.)
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_KDD),np.transpose(Cov_gD))
    if x_pre.shape[0]==1:
        return (mean[0], Cov[0])
    else:
        return (mean, np.diag(Cov))
def gp_evaluate(x,tag,PARAMS,x_data,obs_g):
    if len(x.shape)==1:
        x = np.array([x])
    params_g1,params_g2,inv_KDD1,inv_KDD2 = PARAMS
    if tag == 'g':
        [g1,V_g1] = predict_derivative(x, x_data, obs_g[:,0], params_g1,inv_KDD1,tag="f")
        [g2,V_g2] = predict_derivative(x, x_data, obs_g[:,1], params_g2,inv_KDD2,tag="f")
#        return (np.column_stack([g1,g2])[0], np.column_stack([V_g1,V_g2])[0])
        return (np.array([g1,g2]).T, np.array([V_g1,V_g2]).T)
    elif tag == 'h':
        [h11,V_h11] = predict_derivative(x, x_data, obs_g[:,0], params_g1,inv_KDD1,tag="g1")
        [h12,V_h12] = predict_derivative(x, x_data, obs_g[:,0], params_g1,inv_KDD1,tag="g2")
        [h21,V_h21] = predict_derivative(x, x_data, obs_g[:,1], params_g2,inv_KDD2,tag="g1")
        [h22,V_h22] = predict_derivative(x, x_data, obs_g[:,1], params_g2,inv_KDD2,tag="g2")
#        H = np.row_stack([np.column_stack[h11,h12], np.column_stack[h21,h22]])
#        V_H = np.row_stack([np.column_stack[V_h11,V_h12], np.column_stack[V_h21,V_h22]])
        H = np.array([[h11,h12],[h21,h22]])
        V_H = np.array([[V_h11,V_h12],[V_h21,V_h22]])
        return (H, V_H)
def get_samples_z(x0,v0,PARAMS,x_data,obs_g):
    N= 10
    N_l = 10
    dt = 1e-1
    x = np.repeat(x0[np.newaxis,:],N,axis=0)
    z = x
    v = np.repeat(v0[np.newaxis,:],N,axis=0)
    for j in np.arange(N_l):
        (minus_gradV,V_g) = gp_evaluate(x,'g',PARAMS,x_data,obs_g)
        (minus_H    ,V_h) = gp_evaluate(x,'h',PARAMS,x_data,obs_g)
        
        minus_gradV = np.random.normal(minus_gradV,np.sqrt(V_g))
        minus_H     = np.random.normal(minus_H,np.sqrt(V_h))
        
#        pdb.set_trace()
        f = minus_gradV + 2*np.repeat(np.sum(-minus_gradV * v,axis=1)[:,np.newaxis],2,axis=1)*v
        temp = np.array([np.matmul(v_i[np.newaxis,:],x_i)[0,:] for (x_i,v_i) in zip(np.transpose(minus_H), v)])   
        g = temp + np.repeat(np.sum(v * (-temp),axis=1)[:,np.newaxis],2,axis=1) *v
        x = x + dt*f
        v     = v + dt*g
        z     = np.row_stack([z,x]) 
#    pdb.set_trace()
    return z
def Expected_utility(d,z_prior,PARAMS,x_data,obs_g, K_DD0,K_DD1):
#    pdb.set_trace()
    obs_d_add = np.zeros(d.shape)
    x_data_all = np.row_stack([x_data,d])
    obs_g  = np.row_stack([obs_g, obs_d_add])
    
#    K_Dd = comp_K(x_data,d,PARAMS[0],"f_f")
#    K_dD = K_Dd.T
    K_dd = comp_K(d,d,PARAMS[0],"y_y")
#    inv1_temp = inv_blockMatrix(K_DD0, K_Dd, K_dD, K_dd)
    inv1_temp = np.linalg.inv(K_dd)
#    inv1_temp = np.linalg.inv(comp_K(x_data_all,x_data_all,PARAMS[0],"y_y"))
    
#    pdb.set_trace()
#    K_Dd = comp_K(x_data,d,PARAMS[1],"f_f")
#    K_dD = K_Dd.T
    K_dd = comp_K(d,d,PARAMS[1],"y_y")
#    inv2_temp = inv_blockMatrix(K_DD1, K_Dd, K_dD, K_dd)
    inv2_temp = np.linalg.inv(K_dd)
#    inv2_temp = np.linalg.inv(comp_K(x_data,x_data,PARAMS[1],"y_y"))
    
    PARAMS = (PARAMS[0],PARAMS[1],inv1_temp,inv2_temp)
    (dda,V_g) = gp_evaluate(z_prior,'g',PARAMS,d,obs_d_add)
    (aad,V_h) = gp_evaluate(z_prior,'h',PARAMS,d,obs_d_add)
#    return -(np.mean(np.log(V_g)) + np.mean(np.log(V_g))) 
#    pdb.set_trace()
    alpha1 = .1
    alpha2 = .1
    return -np.mean(np.log((V_g[:,0]+ alpha1 * V_h[0,0,:] + alpha2 * V_h[1,0,:]) * (V_g[:,1]+ alpha1 * V_h[0,1,:]+ alpha2 * V_h[1,1,:] )))

def inv_blockMatrix(p,q,r,s):

  inv_p = np.linalg.inv(p)
  m     = np.linalg.inv(s - np.dot(r, np.dot(inv_p, q)))
  til_p = inv_p + np.mat(inv_p)*np.mat(q)*np.mat(m)*np.mat(r)*np.mat(inv_p)
  til_q = -np.mat(inv_p)*np.mat(q)*np.mat(m)
  til_r = -np.mat(m)*np.mat(r)*np.mat(inv_p)
  til_s = m 
  inv   = np.row_stack((np.column_stack((til_p, til_q)), np.column_stack((til_r, til_s)))) 
  return inv  
  
def DOE(N_add, x0,v,PARAMS,x_data,obs_g,z_prior):
     
#    stime = time.clock()
#    d = np.repeat(x0[np.newaxis,:],N_add,axis=0) + np.random.normal(0,.1,[N_add,D])
#    pdb.set_trace()
    np.random.shuffle(z_prior)
    d0 = z_prior[:N_add,:]
    d = d0
    K_DD0 = comp_K(x_data,x_data,PARAMS[0],"y_y")
    K_DD1 = comp_K(x_data,x_data,PARAMS[1],"y_y")
    Ud_save = []
    # SPSA
    a = .1*1*5
    c=.1*5
    A=100
    alfa=0.602
    gama=0.101  
    for i in np.arange(1,50):
        ak = a/(A+i+1)**alfa
        Ck=c/(i+1)**gama
        # pdb.set_trace()
        # delta=2*ceil(rand(size(d,1),2)-0.5)-1;
        delta = np.round(np.random.rand(d.shape[0],x0.shape[0]))
        d1 = d+Ck*delta
        d2 = d-Ck*delta
        Ud =  Expected_utility(d, z_prior,PARAMS,x_data,obs_g, K_DD0,K_DD1)
        Ud_save.append(Ud)
        Ud1 = Expected_utility(d1,z_prior,PARAMS,x_data,obs_g, K_DD0,K_DD1)
        Ud2 = Expected_utility(d2,z_prior,PARAMS,x_data,obs_g, K_DD0,K_DD1)
        gk=(Ud1-Ud2)/(2*Ck)*delta;
        if (Ud1>Ud) or (Ud2>Ud):
            dnew=d+ak*gk
        else:
            dnew=d
        d=dnew
#    print (time.clock() - stime)
    # pdb.set_trace()
    return d

    
def optimal_paramsAndcompute_invK(x_data,obs_g):
        ########## hyper-parameters optimization-------------
    kernel_u = 1 * RBF(length_scale=0.4, length_scale_bounds=(0.01, 1.5)) \
    + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-4, 1e-2))
    gp1 = GaussianProcessRegressor(kernel=kernel_u,alpha=0.0).fit(x_data,np.array(obs_g)[:,0])
    gp2 = GaussianProcessRegressor(kernel=kernel_u,alpha=0.0).fit(x_data,np.array(obs_g)[:,1])
    params_g1 = np.exp(gp1.kernel_.theta)
    params_g2 = np.exp(gp2.kernel_.theta)
    # pre-train inv(K)
    inv_KDD1 = np.linalg.inv(comp_K(x_data,x_data,params_g1,"y_y"))
    inv_KDD2 = np.linalg.inv(comp_K(x_data,x_data,params_g2,"y_y"))
    PARAMS = (params_g1,params_g2,inv_KDD1,inv_KDD2)
    return PARAMS
if __name__ == '__main__':

    """
    from this line, it is the beginning of main process
    """
    ##############get data-------------
    meshsize = 30
    x1 = np.linspace(0,7,meshsize)
    x2 = np.linspace(-4,7,meshsize)
    XX,YY   = np.meshgrid(x1, x2)
    x1_plot_base = XX.reshape(meshsize*meshsize,1)
    x2_plot_base = YY.reshape(meshsize*meshsize,1)
    x_data_temp = np.column_stack([x1_plot_base,x2_plot_base])
    obs_g_plot = []
    for x in x_data_temp:
        obs_g_plot.append(evaluate(x,'g'))
    obs_g_plot = np.array(obs_g_plot)

    dt = 1.0e-2
    eps = 1.0e-5
    
    # initial x0 v0
    x = np.array([5.87, 6.25])
    v = np.array([0, 1])
    
#    x = np.array([0.59, 0.73]) + np.array([0.02, 0.02])
#    v = np.array([1, 0])

    
    
    v = v/np.sqrt(np.dot(v,v))
    
    N_int = 10
    N_add = 10
    D = x.shape[0]
    x_data = np.repeat(x[np.newaxis,:],N_int,axis=0) + np.random.normal(0,.1,[N_int,D])
    obs_g = []
    for x_i in x_data:
        obs_g.append(evaluate(x_i,'g'))
    obs_g = np.array(obs_g)
    PARAMS = optimal_paramsAndcompute_invK(x_data,obs_g)
    
    temp = 0
    E_f = 1.
    x_save = [x]
    T = 10000
    for i in np.arange(T):
        
        (minus_gradV,V_g) = gp_evaluate(x,'g',PARAMS,x_data,obs_g)
        (minus_H    ,V_h) = gp_evaluate(x,'h',PARAMS,x_data,obs_g)
        f = minus_gradV + 2*np.dot(-minus_gradV,v)*v
#        pdb.set_trace()
        g = np.matmul(minus_H,v) + np.dot(v,np.matmul(-minus_H,v))*v
        x_new = x + dt*f
        v_new = v + dt*g
#        pdb.set_trace()
        E_f = np.sqrt(np.dot(f,f)) + np.sqrt(np.dot(g,g))
        if i%100 ==0:
            print('i:',i,'.Error f:', E_f,'NO.design points:',x_data.shape[0])
            print('x_new:', x_new)
        x = x_new
        v = v_new
        v = v/np.sqrt(np.dot(v,v))
        x_save.append(x)
        if (np.max(V_g) > 0.01) or (np.max(V_h) >0.1):
            print ('DOE-----')
            temp = temp+1
            z_prior = get_samples_z(x,v,PARAMS,x_data,obs_g)
            
#            x_add = np.repeat(x[np.newaxis,:],N_add,axis=0) + np.random.normal(0,.5,[N_add,D])
            x_add = DOE(N_add, x,v,PARAMS,x_data,obs_g,z_prior)
            obs_g_add = []
            for x_add_i in x_add:
                obs_g_add.append(evaluate(x_add_i,'g'))
            obs_g_add = np.array(obs_g_add)
            
            x_data = np.row_stack([x_data,x_add])
            obs_g  = np.row_stack([obs_g, obs_g_add])
            PARAMS = optimal_paramsAndcompute_invK(x_data,obs_g)
#            if temp==5:
#                break
            if temp == 1:
                fig, ax = plt.subplots()
                q = ax.quiver(x1, x2, obs_g_plot[:,0].reshape(30,30), obs_g_plot[:,1].reshape(30,30), alpha=0.3, units='x', pivot='tip', width=0.022, scale=1 / 0.015)
                plt.plot(z_prior[:,0],z_prior[:,1],'g.',markersize=9)
                plt.plot(x_add[:,0],x_add[:,1],'r*',markersize=9)
                plt.plot(np.array(x_save)[:,0],np.array(x_save)[:,1],'k.',markersize=2)
                plt.plot([5.87],[6.25],'k^',markersize=10)
                plt.legend([r'$z$',r'$ d $',r'$x$','LM'], fontsize=20)
                plt.yticks(fontsize=20)
                plt.xticks(fontsize=20)
                plt.xlim([4,7])
                plt.ylim([3,7])
                plt.savefig('example1_x_trace_'+ str(temp) +'.jpg',dpi = 300)
                plt.close()
            else:
                fig, ax = plt.subplots()
                q = ax.quiver(x1, x2, obs_g_plot[:,0].reshape(30,30), obs_g_plot[:,1].reshape(30,30), alpha=0.8, units='x', pivot='tip', width=0.022, scale=1 / 0.015)
                plt.plot(z_prior[:,0],z_prior[:,1],'g.')
                plt.plot(x_add[:,0],x_add[:,1],'r*',markersize=4)
                plt.plot(x_data[:-N_add,0],x_data[:-N_add,1],'r*',markersize=4,alpha=0.4)
                plt.plot(np.array(x_save)[:,0],np.array(x_save)[:,1],'k.',markersize=2)
                plt.plot([5.87,0.59],[6.25,0.76],'k^',markersize=10)
                plt.plot(1.79,3.3,'b^',markersize=10)
                plt.legend([r'$z$',r'$ d $',r'$\mathcal{D}$',r'$x$','LM','SP'], fontsize=10)
                plt.yticks(fontsize=20)
                plt.xticks(fontsize=20)
                plt.savefig('example1_x_trace_'+ str(temp) +'.jpg',dpi = 300)
                plt.close()
        if E_f<eps:
            break
    print('-------------------------')
    print('x_ned:',x)
    print('Amount of xdata:',x_data.shape[0])
    x_plot = np.array(x_save)
#    pdb.set_trace()
    fig, ax = plt.subplots()
    q = ax.quiver(x1, x2, obs_g_plot[:,0].reshape(30,30), obs_g_plot[:,1].reshape(30,30), alpha=0.8, units='x', pivot='tip', width=0.022, scale=1 / 0.015)

    plt.plot(x_data[:,0],x_data[:,1],'r*',markersize=4)
    plt.plot(x_plot[:,0],x_plot[:,1],'k.',markersize=2)
    plt.plot([5.87,0.59],[6.25,0.76],'k^',markersize=10)
#    plt.plot(5.87,6.25,'b',marker='o',markersize=4)
    plt.plot(1.79,3.3,'b^',markersize=10)
    plt.legend([r'$\mathcal{D}$',r'$x$','LM','SP'], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('example1_x_trace_i.jpg',dpi = 300)
    plt.close()
