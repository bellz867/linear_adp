import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import os
import datetime

np.random.seed()

# basis
def phi(X=np.array(2)):
    # [x^2,x*xD,xD^2]
    phi_ = np.array([X[0]**2,X[0]*X[1],X[1]**2])
    return phi_

# basis gradient
def grad_phi(X=np.array(2)):
    #[[2x,0],
    # [xD,x],
    # [0,2xD]]
    grad_phi_ = np.array([[2*X[0],0.0],[X[1],X[0]],[0.0,2*X[1]]])
    return grad_phi_

# def f(Q,R,nu,Gamma,A,B,X,Wa,Wc,GR,XE):


# #classic rk4 method
# def rk4(self,dt,t,X):
#     k1,u1 = f(t,X)
#     k2,u2 = f(t+0.5*dt,X+0.5*dt*k1)
#     k3,u3 = f(t+0.5*dt,X+0.5*dt*k2)
#     k4,u4 = f(t+dt,X+dt*k3)
#     XD = (1.0/6.0)*(k1+2.0*k2+2.0*k3+k4)
#     um = (1.0/6.0)*(u1+2.0*u2+2.0*u3+u4)

#     return XD,um

#bellman error extrapolation
def BEE(Q,R,nu,Gamma,A,B,X,Wa,Wc,GR):
    phi_ = phi(X)
    grad_phi_ = grad_phi(X)
    Gphi = grad_phi_@GR@grad_phi_.T
    u = -0.5/R*np.dot(B,grad_phi_.T@Wa)
    XD = A@X + B*u
    omega = grad_phi_@XD
    rho = 1.0+nu*np.dot(omega,Gamma@omega)
    r = np.dot(X,Q@X)+R*u**2
    delta = r + np.dot(Wc,omega)
    return u,omega,rho,delta,Gphi,phi_

# test using mass spring damper
if __name__ == '__main__':
    dt = 0.01 # time step
    tf = 15.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt))
    steps = int(tf//dt)
    m = 10.0 # mass
    c = 1.0 #damper
    k = 5.0 #spring
    X = np.zeros((2,len(t)))
    X[:,0] = np.array([[1.0,0.0]]) # position velocity
    u = np.zeros_like(t) #inputs
    A = np.array([[0.0,1.0],[-k/m,-c/m]]) # drift
    B = np.array([0,1.0/m]) # control effectiveness
    Wc = np.zeros((3,len(t))) # critic weights
    Wc[:,0] = 0.01*np.random.randn(3)
    Wa = np.zeros((3,len(t))) # actor weights
    Wa[:,0] = 0.01*np.random.randn(3)
    Gamma = 100.0*np.eye(3)
    etac_1 = 0.005 # gain on critic on trajectory
    etac_2 = 0.1 # gain on critic off trajectory
    etaa_1 = 10.0 # gain on difference between Wa and Wc
    etaa_2 = 0.1 # gain on Wa
    lam = 0.4 # gain for Gamma on Gamma
    nu = 0.005
    # Q = np.diag([10.0,5.0]) # state cost
    Q = np.diag([0.1,0.01]) # state cost
    R = 0.001 # input cost
    GR = 1.0/R*np.outer(B,B)
    N = 100 # number of extrapolation points
    md_mag = 0.1 # matched disturbance magnitude
    
    for ii in range(steps):
        X_ii = X[:,ii]
        Wc_ii = Wc[:,ii]
        Wa_ii = Wa[:,ii]
        u_ii,omega_ii,rho_ii,delta_ii,Gphi_ii,phi_ii = BEE(Q,R,nu,Gamma,A,B,X_ii,Wa_ii,Wc_ii,GR)
        XD_ii = A@X_ii + B*(u_ii+md_mag*np.random.randn(1))

        # extrapolations
        Wc_BE = etac_1*delta_ii/rho_ii*omega_ii
        Wa_BE = etac_1/(4.0*rho_ii)*Gphi_ii.T@np.outer(Wa_ii,omega_ii)@Wc_ii
        Gamma_BE = etac_1/(rho_ii**2)*np.outer(omega_ii,omega_ii)

        for jj in range(N):
            X_jj = X_ii + 2.0*np.random.randn(2) # random point around X
            _,omega_jj,rho_jj,delta_jj,Gphi_jj,_ = BEE(Q,R,nu,Gamma,A,B,X_jj,Wa_ii,Wc_ii,GR)
            Wc_BE += etac_2*delta_jj/rho_jj*omega_jj
            Wa_BE += etac_2/(4.0*rho_jj)*Gphi_jj.T@np.outer(Wa_ii,omega_jj)@Wc_ii
            Gamma_BE += etac_2/(rho_jj**2)*np.outer(omega_jj,omega_jj)

        # Wc_BE = delta_ii/rho_ii**2*omega_ii
        # Wa_BE = 1.0/(4.0*rho_ii**2)*Gphi_ii.T@np.outer(Wa_ii,omega_ii)@Wc_ii
        # Gamma_BE = 1.0/(rho_ii**2)*np.outer(omega_ii,omega_ii)

        # for jj in range(N):
        #     X_jj = X_ii + 0.1*np.random.randn(2) # random point around X
        #     _,omega_jj,rho_jj,delta_jj,Gphi_jj = BEE(Q,R,nu,Gamma,A,B,X_jj,Wa_ii,Wc_ii,GR)
        #     Wc_BE += delta_jj/rho_jj**2*omega_jj
        #     Wa_BE += 1.0/(4.0*rho_jj**2)*Gphi_jj.T@np.outer(Wa_ii,omega_jj)@Wc_ii
        #     Gamma_BE += 1.0/(rho_jj**2)*np.outer(omega_jj,omega_jj)

        WcD_ii = -1.0/(N+1)*Gamma@Wc_BE
        WaD_ii = -etaa_1*(Wa_ii-Wc_ii)-etaa_2*Wa_ii+1.0/(N+1)*Wa_BE
        # WaD_ii = -etaa_1*(Wa_ii-Wc_ii)
        GammaD = lam*Gamma-1.0/(N+1)*Gamma@Gamma_BE@Gamma
        if t[ii] > 10:
            b=2

        X[:,ii+1] = X_ii + XD_ii*dt
        Wc[:,ii+1] = Wc_ii + WcD_ii*dt
        Wa[:,ii+1] = Wa_ii + WaD_ii*dt
        u[ii] = u_ii
        Gamma += GammaD*dt
        # Gamma = np.clip(Gamma,0.01,100.0)
        print("t: "+str(round(t[ii],3))+" X "+str(np.round(X[:,ii],3))+" V "+str(np.round(np.dot(Wc_ii,phi_ii),3)))
    X_f = X[:,-1]
    phi_f = phi(X_f)
    grad_phi_f = grad_phi(X_f)
    Wa_f = Wa[:,-1]
    u_f = -0.5/R*np.dot(B,grad_phi_f.T@Wa_f)
    u[-1] = u_f

    # plot the data
    now = datetime.datetime.now()
    nownew = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = "sim-"+nownew
    os.mkdir(path)

    #plot the position
    positionplot,positionax = plot.subplots()
    positionax.plot(t,X[0,:],color='orange',linewidth=2)
    positionax.set_xlabel("$t$ $(sec)$")
    positionax.set_ylabel("$x$ $(m)$")
    positionax.set_title("Position")
    positionax.grid()
    positionplot.savefig(path+"/position.png")

    #plot the velocity
    velocityplot,velocityax = plot.subplots()
    velocityax.plot(t,X[1,:],color='orange',linewidth=2)
    velocityax.set_xlabel("$t$ $(sec)$")
    velocityax.set_ylabel("$\dot{x}$ $(m/s)$")
    velocityax.set_title("Velocity")
    velocityax.grid()
    velocityplot.savefig(path+"/velocity.png")

    #plot the input
    inputplot,inputax = plot.subplots()
    inputax.plot(t,u,color='orange',linewidth=2)
    inputax.set_xlabel("$t$ $(sec)$")
    inputax.set_ylabel("$f_{i}$ $(N)$")
    inputax.set_title("Input")
    inputax.grid()
    inputplot.savefig(path+"/input.png")

    #plot the weights
    acplot,acax = plot.subplots()
    acax.plot(t,Wa[0,:],color='red',linewidth=2,linestyle='dashed')
    acax.plot(t,Wa[1,:],color='green',linewidth=2,linestyle='dashed')
    acax.plot(t,Wa[2,:],color='blue',linewidth=2,linestyle='dashed')
    acax.plot(t,Wc[0,:],color='red',linewidth=2)
    acax.plot(t,Wc[1,:],color='green',linewidth=2)
    acax.plot(t,Wc[2,:],color='blue',linewidth=2)
    acax.set_xlabel("$t$ $(sec)$")
    acax.set_ylabel("$W$ $(m)$")
    acax.set_title("Actor (dashed) Critic (solid) Weights")
    acax.grid()
    acplot.savefig(path+"/weights.png")

    print("Gamma: "+str(np.round(Gamma,3)))
