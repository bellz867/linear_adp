import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import os
import datetime
from scipy.linalg import solve_continuous_are

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
    X = np.zeros((2,len(t)))
    X[:,0] = np.array([[1.0,0.0]]) # position velocity
    u = np.zeros_like(t) #inputs
    A = np.array([[0.0,1.0],[-k/m,-c/m]]) # drift
    B = np.array([[0],[1.0/m]]) # control effectiveness
    Q = np.diag([0.1,0.01]) # state cost
    R = 0.001 # input cost
    P = solve_continuous_are(A,B,Q,np.array([R]))
    K = 1.0/R*(B.T@P)

    md_mag = 0.1 # matched disturbance magnitude
    
    for ii in range(steps):
        X_ii = X[:,ii]
        u_ii = -K@X_ii
        fx = A@X_ii
        bu = B@(u_ii+md_mag*np.random.randn(1))
        XD_ii = fx + bu
        print(fx)
        print(bu)

        X[:,ii+1] = X_ii + XD_ii*dt
        u[ii] = u_ii
        print("t: "+str(round(t[ii],3))+" X "+str(np.round(X[:,ii],3))+" u "+str(np.round(u_ii,3)))
    X_f = X[:,-1]
    u_f = -K@X_f
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