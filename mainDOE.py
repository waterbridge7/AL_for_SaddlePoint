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
from sklearn.gaussian_process.kernels import RBF, WhiteKernel,ConstantKernel
from sklearn.neighbors import KernelDensity
from scipy.linalg import lstsq
import time, dill

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
    D = np.array([[0.8, -0.2],[-0.2,0.5]])
    gamma0 = gamma(x)[0]
    gamma1 = gamma(x)[1]
    if tag == 'f':
        
        w = 0.5*np.matmul(np.matmul(x.T,D),x) - sigma2/2*(np.arctan(x[0] - 5.) + np.arctan(x[1] - 5.))
        return  w #+ np.random.normal(0,.05) # SD
    elif tag == 'g':
         D = np.array([[-1,1],[1,-1]])
         minus_gradV0 = 0.5*sigma2/gamma0 - beta1*x[0] + kpa*(D[0,0]*x[0] + D[0,1]*x[1])
         minus_gradV1 = 0.5*sigma2/gamma1 - beta2*x[1] + kpa*(D[1,0]*x[0] + D[1,1]*x[1])
         return 1*np.array([minus_gradV0,minus_gradV1])
    elif tag == 'h':
        minus_H = np.zeros([2,2])
        minus_H[0,0] = -sigma2*(x[0] -5.)/gamma0**2 - beta1 + kpa*D[0,0]
#        pdb.set_trace()
        minus_H[0,1] = kpa*D[0,1]
        minus_H[1,0] = kpa*D[1,0]
        minus_H[1,1] = -sigma2*(x[1] - 5.)/gamma1**2 - beta2 + kpa*D[1,1]
        return 1*minus_H

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
  elif tag == "f_y":
    K = alpha* K_ff
  elif tag == "f_f":
    K = alpha* K_ff 
  elif tag == "g1_f":
    K = -(dist0)/length_scale**2* alpha * K_ff
  elif tag == "g2_f":
    K = -(dist1)/length_scale**2* alpha * K_ff
  elif tag == "h1_f":
    K = (dist0**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "h2_f":
    K = (dist1**2/length_scale**4 - 1/length_scale**2)* alpha * K_ff
  elif tag == "g1g2_f":
    K = (dist0*dist1/length_scale**4)* alpha * K_ff
  elif tag == "g2g1_f":
    K = (dist0*dist1/length_scale**4)* alpha * K_ff
  elif tag == "g1_g1":
    K = (-dist0**2/length_scale**4 + 1/(length_scale**2))*alpha*K_ff 
  elif tag == "g2_g2":
    K = (-dist1**2/length_scale**4 + 1/(length_scale**2))*alpha*K_ff 
  elif tag == "h1_h1":
    K =  (3/length_scale**4 - 6*dist0**2/length_scale**4 + dist0**4/length_scale**8)*alpha*K_ff
  elif tag == "h2_h2":
    K =  (3/length_scale**4 - 6*dist1**2/length_scale**4 + dist1**4/length_scale**8)*alpha*K_ff
  elif tag == "g1g2_g1g2":
    K =  (1/length_scale**4 - (dist0**2 + dist1**2)/length_scale**6 + dist0**2*dist1**2/length_scale**8)* alpha * K_ff
  elif tag == "g2g1_g2g1":
    K =  (1/length_scale**4 - (dist0**2 + dist1**2)/length_scale**6 + dist0**2*dist1**2/length_scale**8)* alpha * K_ff
  else :
    print("error!")
  return K
def predict_derivative(x_pre, X_data, obs, params,inv_KDD,tag):
  # Data_yAndr include y and r values of the  trainning data
    Data_yAndr = obs
    if tag =="f":
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
    elif tag=="h1":
      Cov_gY = comp_K(x_pre,X_data,params,"h1_f")
      Cov_gD = Cov_gY
      Cov_gg = comp_K(x_pre, x_pre,params,"h1_h1")

      mean = 0. + np.dot(np.dot(Cov_gD,inv_KDD),Data_yAndr-0.)
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_KDD),np.transpose(Cov_gD))
    elif tag=="h2":
      Cov_gY = comp_K(x_pre,X_data,params,"h2_f")
      Cov_gD = Cov_gY
      Cov_gg = comp_K(x_pre, x_pre,params,"h2_h2")

      mean = 0. + np.dot(np.dot(Cov_gD,inv_KDD),Data_yAndr-0.)
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_KDD),np.transpose(Cov_gD))
    elif (tag=="g1g2") or (tag == "g2g1"):
      Cov_gY = comp_K(x_pre,X_data,params,"g1g2_f")
      Cov_gD = Cov_gY
      Cov_gg = comp_K(x_pre, x_pre,params,"g1g2_g1g2")

      mean = 0. + np.dot(np.dot(Cov_gD,inv_KDD),Data_yAndr-0.)
      Cov  = Cov_gg - np.dot(np.dot(Cov_gD,inv_KDD),np.transpose(Cov_gD))
    if x_pre.shape[0]==1:
        return (mean[0], Cov[0])
    else:
        return (mean, abs(np.diag(Cov)))
#def predict_derivative_cov(x_pre, X_data, obs, params,inv_KDD,tag):
#    Data_yAndr = obs
#
#    Cov_fY = comp_K(x_pre, X_data,params,"f_f")
#    Cov_g1f = comp_K(x_pre,X_data,params,"g1_f")
#    Cov_g2f = comp_K(x_pre,X_data,params,"g2_f")
#    Cov_h1f = comp_K(x_pre,X_data,params,"h1_f")
#    Cov_h2f = comp_K(x_pre,X_data,params,"h2_f")
#    Cov_g1g2f = comp_K(x_pre,X_data,params,"g1g2_f")
#    Cov_g2g1f = Cov_g1g2f
#    Cov_Xf = np.row_stack([Cov_fY, Cov_g1f, Cov_g2f, Cov_h1f, Cov_h2f, Cov_g1g2f, Cov_g2g1f])
#    mean = 0. + np.dot(np.dot(Cov_Xf,inv_KDD),Data_yAndr-0.)
#    
#    Cov_fX = Cov_Xf.T
#    COV = np.zeros((7,7))
#    COV[0,0] = comp_K(x_pre, x_pre,params,"f_f")
#    
#    COV[1,0] = comp_K(x_pre, x_pre,params,"g1_f")
#    COV[1,1] = comp_K(x_pre, x_pre,params,"g1_g1")
#    
#    COV[2,0] = comp_K(x_pre, x_pre,params,"g2_f")
#    COV[2,1] = comp_K(x_pre, x_pre,params,"g2_g1")
#    COV[2,2] = comp_K(x_pre, x_pre,params,"g2_g2")
#    
#    COV[3,0] = comp_K(x_pre, x_pre,params,"h1_f")
#    COV[3,1] = comp_K(x_pre, x_pre,params,"h1_g1")
#    COV[3,2] = comp_K(x_pre, x_pre,params,"h1_g2")
#    COV[3,3] = comp_K(x_pre, x_pre,params,"h1_h1")
#    
#    COV[4,0] = comp_K(x_pre, x_pre,params,"h2_f")
#    COV[4,1] = comp_K(x_pre, x_pre,params,"h2_g1")
#    COV[4,2] = comp_K(x_pre, x_pre,params,"h2_g2")
#    COV[4,3] = comp_K(x_pre, x_pre,params,"h2_h1")
#    COV[4,4] = comp_K(x_pre, x_pre,params,"h2_h2")
#    
#    COV[5,0] = comp_K(x_pre, x_pre,params,"g1g2_f")
#    COV[5,1] = comp_K(x_pre, x_pre,params,"g1g2_g1")
#    COV[5,2] = comp_K(x_pre, x_pre,params,"g1g2_g2")
#    COV[5,3] = comp_K(x_pre, x_pre,params,"g1g2_h1")
#    COV[5,4] = comp_K(x_pre, x_pre,params,"g1g2_h2")
#    COV[5,5] = comp_K(x_pre, x_pre,params,"g1g2_g1g2")
#    
#    
#    
#    if tag =="f":
#        return mean[0]
#    elif tag=="g1":
#        return mean[1]
#    elif tag=="g2":
#        return mean[2]
#    elif tag=="h1":
#        return mean[3]
#    elif tag=="h2":
#        return mean[4]
#    elif (tag=="g1g2") or (tag == "g2g1"):
#        return mean[5]
    
def gp_evaluate(x,tag,PARAMS,x_data,obs_g):
    if len(x.shape)==1:
        x = np.array([x]) # x is 2D 
    params_f,inv_KDD = PARAMS
    if tag == 'g':
        [g1,V_g1] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="g1")
        [g2,V_g2] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="g2")
#        return (np.column_stack([g1,g2])[0], np.column_stack([V_g1,V_g2])[0])
        return (np.array([g1,g2]).T, np.array([V_g1,V_g2]).T)
    elif tag == 'h':
        [h11,V_h11] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="h1")
        [h12,V_h12] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="g1g2")
        [h21,V_h21] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="g2g1")
        [h22,V_h22] = predict_derivative(x, x_data, obs_g, params_f,inv_KDD,tag="h2")
#        pdb.set_trace()
#        H = np.row_stack([np.column_stack[h11,h12], np.column_stack[h21,h22]])
#        V_H = np.row_stack([np.column_stack[V_h11,V_h12], np.column_stack[V_h21,V_h22]])
        H = np.array([[h11,h12],[h21,h22]])
        V_H = np.array([[V_h11,V_h12],[V_h21,V_h22]])
        return (H, V_H)
def get_samples_z(x0,v0,PARAMS,x_data,obs_g):
    N= 20
    N_l = 15
    dt = 1e-2
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
        fixed_t = dt/np.sqrt(f[0]**2 + f[1]**2)
        x = x + fixed_t*f
        v     = v + fixed_t*g
        z     = np.row_stack([z,x]) 
    # pdb.set_trace()
    return z
def fast_Expected_utility(d,z_prior,PARAMS,x_data,obs_g, K_DD, alpha_beta_linear):
#    pdb.set_trace()
    obs_d_add = np.zeros(d.shape[0])
    # x_data_all = np.row_stack([x_data,d])
#    if len(obs_g.shape)==1:
#        obs_g = np.array([obs_g])
    obs_g  = np.hstack([obs_g, obs_d_add])
#    K_Dd = comp_K(x_data,d,PARAMS[0],"f_f")
#    K_dD = K_Dd.T
    K_dd = comp_K(d,d,PARAMS[0],"y_y")
#    inv1_temp = inv_blockMatrix(K_DD0, K_Dd, K_dD, K_dd)
    inv_temp = np.linalg.inv(K_dd)
    PARAMS = (PARAMS[0],inv_temp)
    (dda,V_g) = gp_evaluate(z_prior,'g',PARAMS,d,obs_d_add)
    (aad,V_h) = gp_evaluate(z_prior,'h',PARAMS,d,obs_d_add)
    # return -(np.mean(V_g) + np.mean(V_h))
    # pdb.set_trace()
    gamma1 = alpha_beta_linear[1]/alpha_beta_linear[0]
    gamma2 = alpha_beta_linear[2]/alpha_beta_linear[0]
    return -np.mean(np.log((V_g[:,0]+ gamma1 * V_h[0,0,:] + gamma2 * V_h[1,0,:]) * (V_g[:,1]+ gamma1 * V_h[0,1,:]+ gamma2 * V_h[1,1,:] )))

def Expected_utility(d,z_prior,PARAMS,x_data,obs_g, K_DD, alpha_beta_linear):
#    pdb.set_trace()
    obs_d_add = np.zeros(d.shape[0])
    x_data_all = np.row_stack([x_data,d])
#    if len(obs_g.shape)==1:
#        obs_g = np.array([obs_g])
    obs_g  = np.hstack([obs_g, obs_d_add])
    K_DD0= PARAMS[1]
    K_Dd = comp_K(x_data,d,PARAMS[0],"f_f")
    K_dD = K_Dd.T
    K_dd = comp_K(d,d,PARAMS[0],"y_y")
    inv1_temp = inv_blockMatrix(K_DD0, K_Dd, K_dD, K_dd)
#    inv_temp = np.linalg.inv(K_dd)
    PARAMS = (PARAMS[0],inv1_temp)
    (dda,V_g) = gp_evaluate(z_prior,'g',PARAMS,x_data_all,obs_g)
    (aad,V_h) = gp_evaluate(z_prior,'h',PARAMS,x_data_all,obs_g)
    # return -(np.mean(V_g) + np.mean(V_h))
    # pdb.set_trace()
    gamma1 = alpha_beta_linear[1]/alpha_beta_linear[0]
    gamma2 = alpha_beta_linear[2]/alpha_beta_linear[0]
    return -np.mean(np.log((V_g[:,0]+ gamma1 * V_h[0,0,:] + gamma2 * V_h[1,0,:]) * (V_g[:,1]+ gamma1 * V_h[0,1,:]+ gamma2 * V_h[1,1,:] )))
def inv_blockMatrix(p,q,r,s):

  inv_p = np.linalg.inv(p)
  m     = np.linalg.inv(s - np.dot(r, np.dot(inv_p, q)))
  til_p = inv_p + np.mat(inv_p)*np.mat(q)*np.mat(m)*np.mat(r)*np.mat(inv_p)
  til_q = -np.mat(inv_p)*np.mat(q)*np.mat(m)
  til_r = -np.mat(m)*np.mat(r)*np.mat(inv_p)
  til_s = m 
  inv   = np.row_stack((np.column_stack((til_p, til_q)), np.column_stack((til_r, til_s)))) 
  return inv  
  
def DOE(x0,v,PARAMS,x_data,obs_g,z_prior,alpha_beta_linear,N_add):
#    stime = time.clock()
#    d = np.repeat(x0[np.newaxis,:],N_add,axis=0) + np.random.normal(0,.1,[N_add,D])
    # pdb.set_trace()
    np.random.shuffle(z_prior[:60,:])
    d0 = z_prior[:N_add,:]
    d = d0
    K_DD = comp_K(x_data,x_data,PARAMS[0],"y_y")
    Ud_save = []
    # SPSA
    a = .1*1*.2
    c=.1*1.04
    A=100
    alfa=0.602
    gama=0.101  
    for i in np.arange(1,350):
        ak = a/(A+i+1)**alfa
        Ck=c/(i+1)**gama
        # pdb.set_trace()
        # delta=2*ceil(rand(size(d,1),2)-0.5)-1;
        delta = np.round(np.random.rand(d.shape[0],x0.shape[0]))
        d1 = d+Ck*delta
        d2 = d-Ck*delta
        Ud =  fast_Expected_utility(d, z_prior,PARAMS,x_data,obs_g, K_DD,alpha_beta_linear)
        Ud_save.append(Ud)
        Ud1 = fast_Expected_utility(d1,z_prior,PARAMS,x_data,obs_g, K_DD,alpha_beta_linear)
        Ud2 = fast_Expected_utility(d2,z_prior,PARAMS,x_data,obs_g, K_DD,alpha_beta_linear)
        gk=(Ud1-Ud2)/(2*Ck)*delta;
        if (Ud1>Ud) or (Ud2>Ud):
            dnew=d+ak*gk
        else:
            dnew=d
        d=dnew
    print ('d_move:',d-d0)
    print ("Ud improve:", Ud_save[-1] - Ud_save[0])
    
#    print (time.clock() - stime)
    # pdb.set_trace()
    return d

    
def optimal_paramsAndcompute_invK(x_data,obs_g):
        ########## hyper-parameters optimization-------------
        # ConstantKernel(constant_value=1, constant_value_bounds=(1, 1))
    kernel_u = ConstantKernel() * RBF(length_scale=0.9, length_scale_bounds=(0.02, 1)) \
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-3))

    # kernel_u = RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel_u,alpha=0.0).fit(x_data,np.array(obs_g))
    params_f = np.exp(gp.kernel_.theta)
    print ('params_f', params_f)
    # pre-train inv(K)
    inv_KDD = np.linalg.inv(comp_K(x_data,x_data,params_f,"y_y"))
    PARAMS = (params_f,inv_KDD)
    return PARAMS
def compute_parameter(dt, x_save,minus_dV_save,minus_HV_save):
    # t = x_save.shape
    # pdb.set_trace()
    bb = np.zeros(4)
    A = np.zeros((4,3))
    t=len(x_save)
    tem_dv = minus_dV_save[:,:,t-3:t-1]
    tem_dH = minus_HV_save[:,:,t-3:t-1]
    
    bb[0:2] = (x_save[t-1] - x_save[t-2])
    bb[2:4] = (x_save[t-2] - x_save[t-3])
    A1 = np.column_stack([tem_dv[:,:,0].T, tem_dH[:,:,0].T])
    A2 = np.column_stack([tem_dv[:,:,1].T, tem_dH[:,:,1].T])
    A  = np.row_stack((A1,A2))
    p,res,rnk,s=lstsq(A,bb/dt)
    return p

if __name__ == '__main__':

    """
    from this line, it is the beginning of main process
    """
    ##############get data-------------
#    meshsize = 30
#    x1 = np.linspace(0,7,meshsize)
#    x2 = np.linspace(-4,7,meshsize)
#    XX,YY   = np.meshgrid(x1, x2)
#    x1_plot_base = XX.reshape(meshsize*meshsize,1)
#    x2_plot_base = YY.reshape(meshsize*meshsize,1)
#    x_data = np.column_stack([x1_plot_base,x2_plot_base])
#    obs_g = []
#    for x in x_data:
#        obs_g.append(evaluate(x,'g'))
#    obs_g = np.array(obs_g)

    dt = 1.0e-2 #0.1
    eps = 1.0e-5
    
    # initial x0 v0
#    x = np.array([0.46, 0.69]) +np.array([0.01, 0.01])
#    v = np.array([.1, -.5])

#    x = np.array([2.20, 5.98])
#    v = np.array([0, 1]) 

    x = np.array([5.71, 6.23])
    v = np.array([0, -1])

    v = v/np.sqrt(np.dot(v,v))
    
    N_int = 20
    N_add = 10
    D = x.shape[0]
    np.random.seed(23)
    # np.random.seed(12)
    x_data = np.repeat(x[np.newaxis,:],N_int,axis=0) + np.random.normal(0,.5,[N_int,D])
    obs_g = []
    for x_i in x_data:
        obs_g.append(evaluate(x_i,'f'))
    obs_g = np.array(obs_g)
    PARAMS = optimal_paramsAndcompute_invK(x_data,obs_g)
    
    x_linspace = np.linspace(-1,7,100)
    xx,yy = np.meshgrid(x_linspace,x_linspace)
    obs_g_plot = np.zeros((100,100))
    for i in np.arange(100):
        for j in np.arange(100):
            obs_g_plot[i,j] = evaluate(np.array([xx[i,j],yy[i,j]]),'f')
    
    temp = 0
    E_f = 1.
    x_save = [x]
    prior_z_save = []
    design_save = []
    data_save = []
    x_trace_save = []
    T = 10000
    thereshold = 0.2 #0.2
    minus_dV_save  = np.zeros((1,2,T))
    minus_HV_save = np.zeros((2,2,T))
    for i in np.arange(T):
        
        (minus_gradV,V_g) = gp_evaluate(x,'g',PARAMS,x_data,obs_g)
        (minus_H    ,V_h) = gp_evaluate(x,'h',PARAMS,x_data,obs_g)
#        
        # minus_gradV = -evaluate(x,'g')
        # minus_H    = -evaluate(x,'h')
        # V_g = np.zeros((1,2))
        # V_h = np.zeros((2,2,1))
        
        
        f = minus_gradV + 2*np.dot(-minus_gradV,v)*v
#        pdb.set_trace()
        g = np.matmul(minus_H,v) + np.dot(v,np.matmul(-minus_H,v))*v
        
        minus_dV_save[:,:,i] = minus_gradV
        minus_HV_save[:,:,i] = minus_H
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
        # print ('V_g:'+str(np.max(V_g)) + 'V_h:'+str(np.max(V_h)))
        if i==0:
          alpha_beta_linear = np.array([0,0,0])
        else:
          alpha_beta_linear = abs(compute_parameter(dt,x_save,minus_dV_save,minus_HV_save))
        if (alpha_beta_linear[0]*V_g[:,0]+ alpha_beta_linear[1]* V_h[0,0,:] + alpha_beta_linear[2]* V_h[1,0,:]>thereshold) or \
            (alpha_beta_linear[0]*V_g[:,1]+ alpha_beta_linear[1]* V_h[0,1,:]+ alpha_beta_linear[2]* V_h[1,1,:] >thereshold):
        # if (np.max(V_g) > .8) or (np.max(V_h) >.8): #1
            print ('DOE----'+'V_g:'+str(np.max(V_g)) + 'V_h:'+str(np.max(V_h)))

            temp = temp+1
            z_prior = get_samples_z(x,v,PARAMS,x_data,obs_g)
            
#            x_add = np.repeat(x[np.newaxis,:],N_add,axis=0) + np.random.normal(0,.5,[N_add,D])
            x_add = DOE(x,v,PARAMS,x_data,obs_g,z_prior,alpha_beta_linear,N_add)
            obs_g_add = []
            for x_add_i in x_add:
                obs_g_add.append(evaluate(x_add_i,'f'))
            obs_g_add = np.array(obs_g_add)
            
            x_data = np.row_stack([x_data,x_add])
            obs_g  = np.hstack([obs_g, obs_g_add])
            PARAMS = optimal_paramsAndcompute_invK(x_data,obs_g)
            
            # if temp==1:
            #   print ('break loop')
            #   break
            plt.plot(z_prior[:,0],z_prior[:,1],'g.',markersize=9)
            plt.plot(x_add[:,0],x_add[:,1],'k*',markersize=4)
            plt.plot(x_data[:-N_add,0],x_data[:-N_add,1],'r*',markersize=4)
            plt.plot(np.array(x_save)[:,0],np.array(x_save)[:,1],'k*',markersize=2)
            plt.plot([0.46,2.20,5.71],[0.69,5.98,6.23],'k^',markersize=10)
            plt.plot([3.56,1.28],[6.07,3.44],'b^',markersize=10)
        
            plt.contour(xx,yy,obs_g_plot,20,linestyles='dashed')
            plt.legend(['$d$','$X^*$','$x$','local minimum','SP'])
            plt.xlim([-1,7])
            plt.ylim([-1,7])
            plt.savefig('example2_x_trace_' + str(temp) +'.jpg',dpi = 300)
            plt.close()
            
            prior_z_save.append(z_prior)
            design_save.append(x_add)
            data_save.append(x_data)
            x_trace_save.append(np.array(x_save))
    
        if E_f<eps:
            break
    print('save all variable')
    dill.dump_session('gradient_path1.pkl')

    print('-------------------------')
    print('x_ned:',x)
    print('Amount of xdata:',x_data.shape[0])
    x_plot = np.array(x_save)
    np.save('x_trace_path3.npy',x_plot)
    
#    pdb.set_trace()
    # plt.plot(z_prior[:,0],z_prior[:,1],'g.')
#    plt.plot(x_add[:,0],x_add[:,1],'r^',markersize=1)
    plt.plot(x_data[:,0],x_data[:,1],'r*',markersize=4)
    plt.plot(x_plot[:,0],x_plot[:,1],'k*',markersize=2)
    plt.plot([0.46,2.20,5.71],[0.69,5.98,6.23],'k^',markersize=10)
    plt.plot([3.56,1.28],[6.07,3.44],'b^',markersize=10)

    plt.contour(xx,yy,obs_g_plot,20,linestyles='dashed')
    plt.legend([r'$\mathcal{D}$','$x$','LM','SP'],fontsize=20)
    plt.xlim([-1,7])
    plt.ylim([-1,7])
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.savefig('example2_x_trace_i.jpg',dpi = 300)
    plt.close()
