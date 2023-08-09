import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import os
import datetime
from scipy.linalg import solve_lyapunov
from scipy.linalg import solve_discrete_are

np.random.seed()

# test using mass spring damper
if __name__ == '__main__':
    dt = 0.01 # time step
    tf = 15.0 # final time
    t = np.linspace(0.0,tf,int(tf/dt))
    steps = int(tf//dt)
    m = 10.0 # mass
    c = 1.0 #damper
    k = 5.0 #spring
    X_d = np.zeros((3,len(t)))
    X = np.zeros((2,len(t)))
    X[:,0] = np.array([[1.0,0.0]]) # position velocity
    u = np.zeros_like(t) #inputs
    alpha = 10.0 # e gain
    beta = 2.0 # r gain
    md_mag = 0.1 # matched disturbance magnitude
    lambda_ = 0.11 # gain for disturbance
    
    # loop through the dynamics for the number of steps
    for ii in range(steps):
        # current state
        X_ii = X[:,ii]
        x_ii = X_ii[0]
        x_dot_ii = X_ii[1]

        # current desired state
        X_d_ii = X_d[:,ii]
        x_d_ii = X_d_ii[0]
        x_d_dot_ii = X_d_ii[1]
        x_d_dot_dot_ii = X_d_ii[2]

        # error systems
        e_ii = x_d_ii - x_ii
        e_dot_ii = x_d_dot_ii - x_dot_ii
        r_ii = e_dot_ii + alpha*e_ii
        
        # input design
        u_ii = m*(x_d_dot_dot_ii + alpha*e_dot_ii) + c*x_dot_ii + k*x_ii + e_ii + beta*r_ii + lambda_*np.sign(r_ii)

        # integrate
        x_dot_dot_ii = -k/m*x_ii - c/m*x_dot_ii + u_ii/m + md_mag/m*np.random.randn(1)
        X_dot_ii = np.array([x_dot_ii,x_dot_dot_ii[0]])
        X[:,ii+1] = X_ii + X_dot_ii*dt
        u[ii] = u_ii
        print("t: "+str(round(t[ii],3))+" X "+str(np.round(X[:,ii],3))+" u "+str(np.round(u_ii,3)))
    X_f = X[:,-1]
    x_f = X_f[0]
    x_dot_f = X_f[1]

    # current desired state
    X_d_f = X_d[:,-1]
    x_d_f = X_d_f[0]
    x_d_dot_f = X_d_f[1]
    x_d_dot_dot_f = X_d_f[2]
    e_f = x_d_f - x_f
    e_dot_f = x_d_dot_f - x_dot_f
    r_f = e_dot_f + alpha*e_f
    u_f = m*(x_d_dot_dot_f + alpha*e_dot_f) + c*x_dot_f + k*x_f + e_f + beta*r_f + lambda_*np.sign(r_f)
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