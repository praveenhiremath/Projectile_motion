## Author: Praveenkumar Hiremath (Lund University 9/23/2019.)
from __future__ import division
import numpy as np
import matplotlib
from matplotlib import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math 
from math import *
import io
import os
from pylab import *
from scipy.optimize import leastsq
from matplotlib.pyplot import figure
figure(num=None, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
rcParams['legend.fontsize'] = 10

'''
    Solve the "How hard can I throw the ball"-Project 1(a) in
    the Computational Physics course at Lund University, FYTN03. 
    STUDENT:PRAVEENKUMAR HIREMATH
'''

try:
    os.remove('ycoordinates.dat')
    os.remove('xcoordinates.dat')
    os.remove('zcoordinates.dat')
    os.remove('vycoordinates.dat')
    os.remove('vxcoordinates.dat')
    os.remove('vzcoordinates.dat')
    os.remove('RK2NRycoordinates.dat')
    os.remove('RK2NRxcoordinates.dat')
    os.remove('RK2NRzcoordinates.dat')
    os.remove('RK2NRvycoordinates.dat')
    os.remove('RK2NRvxcoordinates.dat')
    os.remove('RK2NRvzcoordinates.dat')
    os.remove('EulerNRycoordinates.dat')
    os.remove('EulerNRxcoordinates.dat')
    os.remove('EulerNRzcoordinates.dat')
    os.remove('EulerNRvycoordinates.dat')
    os.remove('EulerNRvxcoordinates.dat')
    os.remove('EulerNRvzcoordinates.dat')
except OSError:
    print ('Unwanted Files removed')

def euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m):

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    rho_new=np.zeros([size+1])
    B2_m=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=0.0
    z_new[0]=0.0
    # Euler algorithm
    rho_new[0]=float(rho_0)
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    y0=1E4     #((Kb*T)/(b_m*g))= Taken from G&N textbook
    for i in range(0, N):
        B2_m[i]=((0.5*C*rho_new[i]*A)/b_m)
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F_drag_x=(((B2_m[i]*v_new[i]*vx_new[i]))*h)
        F_drag_y=(((B2_m[i]*v_new[i]*vy_new[i]))*h)
        vx_new[i+1]=vx_new[i]-((F_drag_x)*h)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        vy_new[i+1]=vy_new[i]-(g*h)-((F_drag_y)*h)
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        vz_new[i+1]=vz_new[i]-(((S0_m*vx_new[i]*omega))*h)
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        rho_new[i+1]=rho_new[0]*exp(-next_y/y0)
        if (y_new[i+1]<0.0):
          negative=i+1
          break
#    print negative
    max_rho=np.amax(rho_new)
    max_B2_m=np.amax(B2_m)
#    print max_rho, max_B2_m
    final_height_euler=np.amax(y_new)
    final_range_euler=np.amax(x_new)
    curve=np.amax(z_new)
    return final_height_euler,final_range_euler,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative


def RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m):
    m=1.0
    a=0.5
    b=0.5    # m, a, b chosen from book Numerical Recipes 

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    rho_new=np.zeros([size+1])
    B2_m=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=0.0
    z_new[0]=0.0
    # 2nd Order Runge-Kutta algorithm

    rho_new[0]=float(rho_0)
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    y0=1E4     #((Kb*T)/(b_m*g))
    for i in range(0, N):
        B2_m[i]=((0.5*C*rho_new[i]*A)/b_m)
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F1_drag_x=(((B2_m[i]*v_new[i]*vx_new[i]))*h)
        k1_x=-((F1_drag_x)*h)
        F2_drag_x=((B2_m[i]*v_new[i]*(vx_new[i]+(m*k1_x)))*h)
        k2_x=-((F2_drag_x)*h)
        vx_new[i+1]=vx_new[i]+(a*k1_x)+(b*k2_x)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        F1_drag_y=(((B2_m[i]*v_new[i]*vy_new[i]))*h)
        k1_y=-(g*h)-((F1_drag_y)*h)
        F2_drag_y=((B2_m[i]*v_new[i]*(vy_new[i]+(m*k1_y)))*h)
        k2_y=-(g*h)-((F2_drag_y)*h)
        vy_new[i+1]=vy_new[i]+(a*k1_y)+(b*k2_y)         
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        k1_z=-(((S0_m*vx_new[i]*omega))*h)
        k2_z=-(((S0_m*(vx_new[i]+(m*k1_z))*omega))*h)
        vz_new[i+1]=vz_new[i]+(a*k1_z)+(b*k2_z)      
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        rho_new[i+1]=rho_new[0]*exp(-next_y/y0)
        if (y_new[i+1]<0.0):
          negative=i+1
          break
    max_rho=np.amax(rho_new)
    max_B2_m=np.amax(B2_m)
#    print max_B2_m
#    print max_rho, max_B2_m
    final_height_RK2=np.amax(y_new)
    final_range_RK2=np.amax(x_new)
    return final_height_RK2,final_range_RK2,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative

##### Without Air resistance and other effects
def analytical_rangeR(v_0, angle_rad, initial_height):
  g=9.8 #Gravitational acceleration
  Analytic_rangeR=((pow(v_0,2)*sin(2*angle_rad))/(2*g))*(1+pow((1+((2*g*initial_height)/(pow(v_0,2)*pow(sin(angle_rad),2)))),0.5))
  return Analytic_rangeR

###########GIVEN final velocity and angle and Horizontal range

def euler_reverse_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, vz_0, b_m, h, N, rho_0, S0_m):
    
#    Euler reverse trajectory (Final to Initial position)

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    rho_new=np.zeros([size+1])
    B2_m=np.zeros([size+1])
    x_new[N]=HR
    y_new[N]=0.0
    z_new[N]=ZR
    # Euler algorithm
    rho_new[N]=float(rho_0)
    v_new[N]=float(v_0)
    vx_new[N]=float(vx_0)
    vy_new[N]=float(vy_0)
    vz_new[N]=float(vz_0)
    y0=1E4     #((Kb*T)/(b_m*g))
    for i in range(N, 0, -1):
        B2_m[i]=((0.5*C*rho_new[i]*A)/b_m)
        x_new[i-1]=x_new[i]-(vx_new[i]*h)
        F_drag_x=(((B2_m[i]*v_new[i]*vx_new[i]))*h)
        F_drag_y=(((B2_m[i]*v_new[i]*vy_new[i]))*h)
        vx_new[i-1]=vx_new[i]+((F_drag_x)*h)
        y_new[i-1]=y_new[i]-(vy_new[i]*h)
        vy_new[i-1]=vy_new[i]+(g*h)+((F_drag_y)*h)
        z_new[i-1]=z_new[i]-(vz_new[i]*h)
        vz_new[i-1]=vz_new[i]+(((S0_m*vx_new[i]*omega))*h)
        v_new[i-1]=sqrt(pow(vx_new[i-1],2)+pow(vy_new[i-1],2)+pow(vz_new[i-1],2))
        next_y=y_new[i]
        rho_new[i-1]=rho_new[0]*exp(-next_y/y0)
        if (y_new[i-1]<0.0):
          negative=i-1
          break
    max_rho=np.amax(rho_new)
    max_B2_m=np.amax(B2_m)
#    print max_rho, max_B2_m
    final_height_euler=np.amax(y_new)
    final_range_euler=np.amax(x_new)
    curve=np.amax(z_new)
    return final_height_euler,final_range_euler,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative


def cannon_RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, B2_m, S0_m):
    m=1.0
    a=0.5
    b=0.5         # m, a, b chosen from book Numerical Recipes 

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    rho_new=np.zeros([size+1])
#    B2_m=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=0.0
    z_new[0]=0.0
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    y0=1E4     #((Kb*T)/(b_m*g))
    # 2nd Order Runge-Kutta algorithm
    for i in range(0, N, 1):
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F1_drag_x=(((B2_m*v_new[i]*vx_new[i]))*h)
        k1_x=-((F1_drag_x)*h)
        F2_drag_x=((B2_m*v_new[i]*(vx_new[i]+(m*k1_x)))*h)
        k2_x=-((F2_drag_x)*h)
        vx_new[i+1]=vx_new[i]+(a*k1_x)+(b*k2_x)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        F1_drag_y=(((B2_m*v_new[i]*vy_new[i]))*h)
        k1_y=-(g*h)-((F1_drag_y)*h)
        F2_drag_y=((B2_m*v_new[i]*(vy_new[i]+(m*k1_y)))*h)
        k2_y=-(g*h)-((F2_drag_y)*h)
        vy_new[i+1]=vy_new[i]+(a*k1_y)+(b*k2_y)         
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        k1_z=-(((S0*vx_new[i]*omega)/b_m)*h)
        k2_z=-(((S0*(vx_new[i]+(m*k1_z))*omega)/b_m)*h)
        vz_new[i+1]=vz_new[i]+(a*k1_z)+(b*k2_z)      
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        if (y_new[i+1]<0.0):
          negative=i+1
          break
    max_B2_m=B2_m
    final_height_RK2=np.amax(y_new)
    final_range_RK2=np.amax(x_new)
    return final_height_RK2,final_range_RK2,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative

def cannon_euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0,B2_m):

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=0.0
    z_new[0]=0.0
    # Euler algorithm
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    for i in range(0, N, 1):
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F_drag_x=(((B2_m*v_new[i]*vx_new[i]))*h)
        F_drag_y=(((B2_m*v_new[i]*vy_new[i]))*h)
        vx_new[i+1]=vx_new[i]-((F_drag_x)*h)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        vy_new[i+1]=vy_new[i]-(g*h)-((F_drag_y)*h)
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        vz_new[i+1]=vz_new[i]
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        if (y_new[i+1]<0.0):
          negative=i+1
          break
#    print negative
#    max_rho=np.amax(rho_new)
#    print "DRAG"
#    print F_drag_x, F_drag_y
    max_B2_m=B2_m
#    print max_B2_m
#    print max_rho, max_B2_m
    final_height_euler=np.amax(y_new)
    final_range_euler=np.amax(x_new)
    curve=np.amax(z_new)
    return final_height_euler,final_range_euler,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative


def Baseball_RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, B2_m,S0_m):
    m=1.0
    a=0.5
    b=0.5         # m, a, b chosen from book Numerical Recipes 

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    rho_new=np.zeros([size+1])
#    B2_m=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=1.128      #in meter. 1.128 meter is approximately 3.7 feet
    z_new[0]=0.0
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    # 2nd Order Runge-Kutta algorithm
    for i in range(0, N, 1):
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F1_drag_x=(((B2_m*v_new[i]*vx_new[i]))*h)
        k1_x=-((F1_drag_x)*h)
        F2_drag_x=((B2_m*v_new[i]*(vx_new[i]+(m*k1_x)))*h)
        k2_x=-((F2_drag_x)*h)
        vx_new[i+1]=vx_new[i]+(a*k1_x)+(b*k2_x)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        k1_y=-(g*h)
        k2_y=-(g*h)
        vy_new[i+1]=vy_new[i]+(a*k1_y)+(b*k2_y)         
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        k1_z=-(((S0_m*vx_new[i]*omega))*h)
        k2_z=-(((S0_m*(vx_new[i]+(m*k1_z))*omega))*h)
        vz_new[i+1]=vz_new[i]+(a*k1_z)+(b*k2_z)      
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        if (y_new[i+1]<0.0):
          negative=i+1
          break
    max_B2_m=B2_m
    final_height_RK2=np.amax(y_new)
    final_range_RK2=np.amax(x_new)
    return final_height_RK2,final_range_RK2,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative

def Baseball_euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0,B2_m, S0_m):

    g=9.8    #gravitaional acceleration
    size=N
    vx_new=np.zeros([size+1])
    vy_new=np.zeros([size+1])
    vz_new=np.zeros([size+1])
    v_new=np.zeros([size+1])
    x_new=np.zeros([size+1])
    y_new=np.zeros([size+1])
    z_new=np.zeros([size+1])
    x_new[0]=0.0
    y_new[0]=3.7
    z_new[0]=0.0
    # Euler algorithm
    v_new[0]=float(v_0)
    vx_new[0]=float(vx_0)
    vy_new[0]=float(vy_0)
    vz_new[0]=0.0
    for i in range(0, N):
        x_new[i+1]=x_new[i]+(vx_new[i]*h)
        F_drag_x=(((B2_m*v_new[i]*vx_new[i]))*h)
        print F_drag_x
        vx_new[i+1]=vx_new[i]-((F_drag_x)*h)
        y_new[i+1]=y_new[i]+(vy_new[i]*h)
        vy_new[i+1]=vy_new[i]-(g*h)
        z_new[i+1]=z_new[i]+(vz_new[i]*h)
        vz_new[i+1]=vz_new[i]-(((S0_m*vx_new[i]*omega))*h)
        v_new[i+1]=sqrt(pow(vx_new[i+1],2)+pow(vy_new[i+1],2)+pow(vz_new[i+1],2))
        next_y=y_new[i]
        if (y_new[i+1]<0.0):
          negative=i+1
          break
    max_B2_m=np.amax(B2_m)
#    print max_rho, max_B2_m
    final_height_euler=np.amax(y_new)
    final_range_euler=np.amax(x_new)
    curve=np.amax(z_new)
    return final_height_euler,final_range_euler,x_new,y_new,z_new,vx_new,vy_new,vz_new,negative

print "Note: Please first run task 4\n"

pi=22/7
DO=input("Enter what you want to do\n1. Cannon trajectory in presence of only air resistance, altitude effect and gravity\n2. Effect of air resistance on initial (throwing) angle in cannon (as in book G&N:only air resistance and gravity)\n3. Baseball: curve ball trajectory\n4. Test Euler algorithm with air and other effect and test 2nd Order RK algorithm with air and other effect and compare\n5. Calculate horizontal range analytically using the available expression. Then use 2nd Order RK and Euler for the case without resistance to compare with analytical value for horizontal range X (Validity of code)\n6. Calculate optimal angle for a given initial velocity (considering different influencing factors)\n7. Calculate optimal angle for a given initial velocity (under only G)\n8. Plot the reverse tranjectory when final horizontal, vertical, deviation along z-axis, velocity and angle are known\n9. Finding initial conditions when different final conditions are known\n")
print "Chosen task is =",DO



if (DO==1):
   print "Cannon ball example from textbook and only air resistance, altitude effects are considered 2nd order RK used"
   v_0=700                         # Cannon initial velocity in ms^-1 (100.662 mph)
   angle=40                        # angle in degrees   
   angle_rad=(pi*angle)/180        # initial angle (at t=0) in radians
   vx_0=v_0*cos(angle_rad)         # initial velocity (at t=0) in x_direction
   vy_0=v_0*sin(angle_rad)         # initial velocity (at t=0) in y_direction
   b_m=0.1924                      # Mass of cannon in kg
   B1=0.0                          # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                          # Shape dependent constant (experimentally determined C-values) for sphere (Re approximately < 20E-4)
   rho_0=1.225                     # Density at sea level (approximate)
   A=7.854E-3                      # Frontal area (cannon frontal area in m^2 as diameter is 10cm)
   h1=0.000041                      # step size (timestep) in s
   h=h1
   N=5000000;                       # number of euler steps (number of timesteps)
   T=273                           # Temperature in Kelvin
   Kb=1.38064852E-23               # Boltzman constant
   S0=0.00006109                   # SPIN
   S0_m=0.0                     
   rpm=0                          # revolutions per second
   omega=2*pi*rpm                  # radians per second

   print "Cannon ball Initial velocity =", v_0, "m/s and Initial angle =", angle_rad,"in radians=",angle,"degrees"
   final_height_cannon,final_range_cannon,x_new_cannon,y_new_cannon,z_new_cannon,vx_new_cannon,vy_new_cannon,vz_new_cannon,cannon_neg_index=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)

   x_new_cannon=x_new_cannon[:cannon_neg_index]
   y_new_cannon=y_new_cannon[:cannon_neg_index]
   z_new_cannon=z_new_cannon[:cannon_neg_index]
   vx_new_cannon=vx_new_cannon[:cannon_neg_index]
   vy_new_cannon=vy_new_cannon[:cannon_neg_index]
   vz_new_cannon=vz_new_cannon[:cannon_neg_index]
   cannonxyz=np.zeros(3)
   cannonxyz=np.array([vx_new_cannon[cannon_neg_index-1],vy_new_cannon[cannon_neg_index-1],vz_new_cannon[cannon_neg_index-1]])
   np.savetxt('cannonfinal_vel_components.dat',cannonxyz)

#####    TO SAVE/OPEN COORDINATES AND VELOCITIES INTO FILES BLOCK (UNCOMMENT TO USE)
   np.savetxt('cannonRK2xcoordinates.dat', x_new_cannon)
   np.savetxt('cannonRK2ycoordinates.dat', y_new_cannon)
   np.savetxt('cannonRK2zcoordinates.dat', z_new_cannon)
   np.savetxt('cannonRK2vxcoordinates.dat', vx_new_cannon)
   np.savetxt('cannonRK2vycoordinates.dat', vy_new_cannon)
   np.savetxt('cannonRK2vzcoordinates.dat', vy_new_cannon)

   cannonRK2_x=np.loadtxt('cannonRK2xcoordinates.dat')
   cannonRK2_y=np.loadtxt('cannonRK2ycoordinates.dat')
   cannonRK2_z=np.loadtxt('cannonRK2zcoordinates.dat')

####     TO PLOT TRAJECTORY IN XY/XZ PLANE BLOCK (UNCOMMENT TO USE)
#   fig, ax = plt.subplots()

#   ax.plot(cannonRK2_x,cannonRK2_z,'y--',label='cannonXZ-Plane Runge-Kutta 2nd Order, h=0.000041')
#   ax.plot(cannonRK2_x,cannonRK2_y,'y--',label='cannonXY-Plane Runge-Kutta 2nd Order, h=0.000041')

#   ax.set_title('CANNON PROJECTILE XY PLANE')
#   ax.grid(True)
#   xlabel('X Range (m)')
#   ylabel('Height (m)')
#   ylabel('Distance along Z-axis (m)')
#   legend(loc='best')
#   savefig('cannonXYProjectile_compare.png')

####    TO PLOT TRAJECTORY IN 3D BLOCK (UNCOMMENT TO USE)
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(cannonRK2_x, cannonRK2_y, cannonRK2_z,'r--', label='Cannon trajectory 2nd Order RK, h=0.000041')
   ax.set_xlabel('X (Horizonatal Range (m))')
   ax.set_ylabel('Y (Elevation from ground (m))')
   ax.set_zlabel('Z (distance along magnus force (m))')
   legend(loc='best')
   #print ax.azim
   ax.view_init(azim=10)
   savefig('cannon3DProjectile.png')

   print "Using RK2, cannon final height, cannon final range, cannon z_deflection respectively are:", final_height_cannon,final_range_cannon,z_new_cannon[cannon_neg_index-1]


if (DO==2):
######## DIFFERENT ANGLES  
   print "Cannon ball example from textbook and only air resistance is considered"
   v_0=700                         # Cannon initial velocity in ms^-1 (100.662 mph)
   angle=40                        # angle in degrees   
   angle_rad=(pi*angle)/180        # initial angle (at t=0) in radians
   vx_0=v_0*cos(angle_rad)         # initial velocity (at t=0) in x_direction
   vy_0=v_0*sin(angle_rad)         # initial velocity (at t=0) in y_direction
   b_m=0.1924                      # Mass of cannon in kg
   B1=0.0                          # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=1.00                          # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225                     # Density at sea level (approximate)
   A=7.854E-3                      # Frontal area (Baseball frontal area in m^2=0.004417865)
   d=0.1                           # diameter of cannon
   B2_m=1E-1                       # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4)
   h1=0.00041                      # step size (timestep) in s
   h=h1
   N=500000;                       # number of euler steps (number of timesteps)
   T=273                           # Temperature in Kelvin
   Kb=1.38064852E-23               # Boltzman constant
   S0=0.00006109                   # SPIN
   S0_m=0.0
   rpm=0                           # revolutions per second
   omega=2*pi*rpm                  # radians per second

   print "Cannon ball Initial velocity =", v_0, "m/s and Initial angle =", angle_rad,"in radians=",angle,"degrees"

   Num_angle_iter=7
   angles_degree=np.linspace(30,60,Num_angle_iter)
   angles_radians=(pi*angles_degree)/180

   final_height_cannonRK2=np.zeros(Num_angle_iter)
   final_range_cannonRK2=np.zeros(Num_angle_iter)
   all_xranges_cannonRK2=np.zeros(Num_angle_iter)
   cannonneg_index=np.zeros(Num_angle_iter)
   for i in range(0,Num_angle_iter,1):
     vx_0=v_0*cos(angles_radians[i])
     vy_0=v_0*sin(angles_radians[i])
     final_height_cannonRK2[i],final_range_cannonRK2[i],x_newcannon,y_newcannon,z_newcannon,vx_newcannon,vy_newcannon,vz_newcannon,cannonneg_index[i]=cannon_RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, B2_m, S0_m)  
     all_xranges_cannonRK2[i]=final_range_cannonRK2[i]
     #print angles_degree[i], final_range_cannonRK2[i]
     temp=int(cannonneg_index[i])
     #print temp
     x_newcannon=x_newcannon[:temp]
     x_newcannon=x_newcannon/1000
     y_newcannon=y_newcannon[:temp]
     y_newcannon=y_newcannon/1000
     z_newcannon=z_newcannon[:temp]
     z_newcannon=z_newcannon/1000
#####   ANGLE DEPENDENCE IN PRESENCE OF OTHER EFFECTS + AIR RESISTANCE PLOT BLOCK (UNCOMMENT TO USE) 
     plt.plot(x_newcannon, y_newcannon, label=angles_degree[i]) 
     plt.axis([0, 55, 0, 25])
     plt.xlabel('X Range (km)')
     plt.ylabel('Height (km)')
     plt.grid(True)
     plt.title('CANNON WITH A GIVEN VELOCITY (Without air drag)\nAT DIFFERENT ANGLES (DEGREES) XY PLANE (RK2)')
     legend(loc='best',prop={'size': 8})
     savefig('CANNON_XYRK_Angles_No_factor_projectile.png')
   
   for i in range(1,Num_angle_iter,1):
     index_max_x=argmax(all_xranges_cannonRK2)
     Opt_angle= angles_degree[index_max_x]
   #  print "Optimal initial angle (degree) of initial velocity =" 
   print "Optimal angle and xrange"
   print Opt_angle, all_xranges_cannonRK2[index_max_x]

if (DO==3):
   print "Baseball: Trajectory of side arm curve ball"
   v_0=31.2928                     # Baseball initial velocity in ms^-1 (70 mph)
   angle=0.0                       # angle=0.0 degrees because the ball is thrown with an initial velocity in x-direction   
   angle_rad=(pi*angle)/180        # initial angle (at t=0) in radians
   vx_0=v_0*cos(angle_rad)         # initial velocity (at t=0) in x_direction
   vy_0=v_0*sin(angle_rad)         # initial velocity (at t=0) in y_direction
   b_m=0.149                       # Mass of baseball in kg
   B1=0.0                          # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                          # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225                     # Density at sea level (approximate)
   A=4.5365E-3                     # Frontal area (Baseball frontal area in m^2=0.004417865)
   d=0.076                         # diameter of baseball
   B2_m=8.76465E-3                 # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4)
   h1=0.000041                      # step size (timestep) in s
   h=h1
   N=500000;                       # number of euler steps (number of timesteps)
   T=273                           # Temperature in Kelvin
   Kb=1.38064852E-23               # Boltzman constant
   S0=0.06109                      # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=4.1E-4
   rpm=30                          # revolutions per second
   omega=2*pi*rpm                  # radians per second

   print "Cannon ball Initial velocity =", v_0, "m/s and Initial angle =", angle_rad,"in radians=",angle,"degrees"
   print "angle=0.0 degrees because the ball is thrown with an initial velocity in x-direction"

   final_height_BaseRK2,final_range_BaseRK2,x_new_BaseRK2,y_new_BaseRK2,z_new_BaseRK2,vx_new_BaseRK2,vy_new_BaseRK2,vz_new_BaseRK2,BaseRK2_neg_index=Baseball_RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, B2_m, S0_m)

   #final_height_BaseRK2,final_range_BaseRK2,x_new_BaseRK2,y_new_BaseRK2,z_new_BaseRK2,vx_new_BaseRK2,vy_new_BaseRK2,vz_new_BaseRK2,BaseRK2_neg_index=Baseball_euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, B2_m, S0_m)

   x_new_BaseRK2=x_new_BaseRK2[:BaseRK2_neg_index]
   y_new_BaseRK2=y_new_BaseRK2[:BaseRK2_neg_index]
   z_new_BaseRK2=z_new_BaseRK2[:BaseRK2_neg_index]
   vx_new_BaseRK2=vx_new_BaseRK2[:BaseRK2_neg_index]
   vy_new_BaseRK2=vy_new_BaseRK2[:BaseRK2_neg_index]
   vz_new_BaseRK2=vz_new_BaseRK2[:BaseRK2_neg_index]
   Basetempxyz=np.zeros(3)
   Basetempxyz=np.array([vx_new_BaseRK2[BaseRK2_neg_index-1],vy_new_BaseRK2[BaseRK2_neg_index-1],vz_new_BaseRK2[BaseRK2_neg_index-1]])
   #np.savetxt('baseball_final_vel_components.dat',tempxyz)


#####    TO SAVE/OPEN COORDINATES AND VELOCITIES INTO FILES BLOCK (UNCOMMENT TO USE)
   np.savetxt('BaseRK2xcoordinates.dat', x_new_BaseRK2)
   np.savetxt('BaseRK2ycoordinates.dat', y_new_BaseRK2)
   np.savetxt('BaseRK2zcoordinates.dat', z_new_BaseRK2)
   np.savetxt('BaseRK2vxcoordinates.dat', vx_new_BaseRK2)
   np.savetxt('BaseRK2vycoordinates.dat', vy_new_BaseRK2)
   np.savetxt('BaseRK2vzcoordinates.dat', vy_new_BaseRK2)

   BaseRK2_x=np.loadtxt('BaseRK2xcoordinates.dat')
   BaseRK2_y=np.loadtxt('BaseRK2ycoordinates.dat')
   BaseRK2_z=np.loadtxt('BaseRK2zcoordinates.dat')

####     TO PLOT TRAJECTORY IN XY/XZ PLANE BLOCK (UNCOMMENT TO USE)
   #fig, ax = plt.subplots()
   #ax.plot(BaseRK2_x,BaseRK2_z,'r-x',label='Baseball XZ-Plane Runge-Kutta 2nd Order, h=0.000041')
   #ax.plot(BaseRK2_x,BaseRK2_y,'r-x',label='Baseball XY-Plane Runge-Kutta 2nd Order, h=0.000041')
   #ax.set_title('BASEBALL TRAJECTORY XY PLANE')
   #ax.grid(True)
   #xlabel('X Range (m)')
   #ylabel('Height (m)')
   #ylabel('Distance along Z-axis (m)')
   #legend(loc='best')
   #savefig('BaseballXYProjectile_compare.png')

####    TO PLOT TRAJECTORY IN 3D BLOCK (UNCOMMENT TO USE)
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(BaseRK2_x, BaseRK2_y, BaseRK2_z,'r-x', label='Baseball trajectory 2nd Order RK, h=0.000041')
   ax.set_xlabel('X (Horizonatal Range (m))')
   ax.set_ylabel('Y (Elevation from ground (m))')
   ax.set_zlabel('Z (distance along magnus force (m))')
   legend(loc='best')
   #print ax.azim
   ax.view_init(azim=10)
   savefig('Baseball3DProjectile.png')

   print "Using RK2, Baseball Initial height, final horizontal range, z_deflection respectively are (in m):",final_height_BaseRK2, final_range_BaseRK2,z_new_BaseRK2[BaseRK2_neg_index-1]

   
   #print "Using Euler, Baseball Initial height, final horizontal range, z_deflection respectively are (in m):",final_height_BaseRK2, final_range_BaseRK2,z_new_BaseRK2[BaseRK2_neg_index-1]

###### Comparison of Euler and RK2 performance
if (DO==4):
   v_0=45                   # initial velocity in ms^-1 (100.662 mph)
   angle=40                 # angle in degrees   
   angle_rad=(pi*angle)/180 # initial angle (at t=0) in radians
   vx_0=v_0*cos(angle_rad)  # initial velocity (at t=0) in x_direction
   vy_0=v_0*sin(angle_rad)  # initial velocity (at t=0) in y_direction
   b_m=0.149                # Mass of baseball in kg
   B1=0.0                   # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                   # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225              # Density at sea level (approximate)
   A=4.5365E-3              # Frontal area (Baseball diameter = 7.6cm)
   d=0.076                  # diameter of baseball
   B2_m=8.76465E-3          # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4)
   h1=0.000041              # step size (timestep) in s
   h=h1
   N=500000;                # number of euler steps (number of timesteps)
   T=273                    # Temperature in Kelvin
   Kb=1.38064852E-23        # Boltzman constant
   S0=0.06109               # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=4.1E-4              # Taken from textbook (G&N) for baseball.
   rpm=30                   # revolutions per second
   omega=2*pi*rpm           # radians per second

   print "Initial velocity =", v_0, "m/s and Initial angle =", angle_rad,"in radians=",angle,"degrees"

   final_height_euler,final_range_euler,x_new_euler,y_new_euler,z_new_euler,vx_new_euler,vy_new_euler,vz_new_euler,euler_neg_index=euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)
   x_new_euler=x_new_euler[:euler_neg_index]
   y_new_euler=y_new_euler[:euler_neg_index]
   z_new_euler=z_new_euler[:euler_neg_index]

#####    TO SAVE/OPEN COORDINATES AND VELOCITIES INTO FILES BLOCK (UNCOMMENT TO USE)
   np.savetxt('xcoordinates.dat', x_new_euler)
   np.savetxt('ycoordinates.dat', y_new_euler)
   np.savetxt('zcoordinates.dat', z_new_euler)
   np.savetxt('vxcoordinates.dat', vx_new_euler)
   np.savetxt('vycoordinates.dat', vy_new_euler)
   np.savetxt('vzcoordinates.dat', vy_new_euler)

   print "Using euler, Final height, final range, z_deflection respectively are:", final_height_euler,final_range_euler,z_new_euler[euler_neg_index-1]

   h2=0.000041
   h=h2
   final_height_RK2,final_range_RK2,x_new_RK2,y_new_RK2,z_new_RK2,vx_new_RK2,vy_new_RK2,vz_new_RK2,RK2_neg_index=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)

   x_new_RK2=x_new_RK2[:RK2_neg_index]
   y_new_RK2=y_new_RK2[:RK2_neg_index]
   z_new_RK2=z_new_RK2[:RK2_neg_index]
   vx_new_RK2=vx_new_RK2[:RK2_neg_index]
   vy_new_RK2=vy_new_RK2[:RK2_neg_index]
   vz_new_RK2=vz_new_RK2[:RK2_neg_index]
   tempxyz=np.zeros(3)
   tempxyz=np.array([vx_new_RK2[RK2_neg_index-1],vy_new_RK2[RK2_neg_index-1],vz_new_RK2[RK2_neg_index-1]])
   np.savetxt('final_vel_components.dat',tempxyz) 


#####    TO SAVE/OPEN COORDINATES AND VELOCITIES INTO FILES BLOCK (UNCOMMENT TO USE)
   np.savetxt('RK2xcoordinates.dat', x_new_RK2)
   np.savetxt('RK2ycoordinates.dat', y_new_RK2)
   np.savetxt('RK2zcoordinates.dat', z_new_RK2)
   np.savetxt('RK2vxcoordinates.dat', vx_new_RK2)
   np.savetxt('RK2vycoordinates.dat', vy_new_RK2)
   np.savetxt('RK2vzcoordinates.dat', vy_new_RK2)

   euler_x=np.loadtxt('xcoordinates.dat')
   euler_y=np.loadtxt('ycoordinates.dat')
   euler_z=np.loadtxt('zcoordinates.dat')
   RK2_x=np.loadtxt('RK2xcoordinates.dat')
   RK2_y=np.loadtxt('RK2ycoordinates.dat')
   RK2_z=np.loadtxt('RK2zcoordinates.dat')

####     TO PLOT TRAJECTORY IN XY/XZ PLANE BLOCK (UNCOMMENT TO USE)
   #fig, ax = plt.subplots()
   #ax.plot(euler_x,euler_z,'r-x',label='XZ-Plane Euler Forward, h=0.000041')
   #ax.plot(RK2_x,RK2_z,'y--',label='XZ-Plane Runge-Kutta 2nd Order, h=0.000041')
   #ax.plot(euler_x,euler_y,'r-x',label='XY-Plane Euler Forward, h=0.000041')
   #ax.plot(RK2_x,RK2_y,'y--',label='XY-Plane Runge-Kutta 2nd Order, h=0.000041')
   #ax.set_title('PROJECTILE XY PLANE')
   #ax.grid(True)
   #xlabel('X Range (m)')
   #ylabel('Height (m)')
   #ylabel('Distance along Z-axis (m)')
   #legend(loc='best')
   #savefig('XYProjectile_compare.png')

####    TO PLOT TRAJECTORY IN 3D BLOCK (UNCOMMENT TO USE)
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(euler_x, euler_y, euler_z, 'r-x', label='Ball trajectory Euler, h=0.000041')
   ax.plot(RK2_x, RK2_y, RK2_z,'y--', label='Ball trajectory 2nd Order RK, h=0.000041')
   ax.set_xlabel('X (Horizonatal Range (m))')
   ax.set_ylabel('Y (Elevation from ground (m))')
   ax.set_zlabel('Z (distance along magnus force (m))')
   legend(loc='best')
   ax.view_init(azim=10)
   savefig('3DProjectile.png')

   print "Using RK2, Final height, final range, z_deflection respectively are:", final_height_RK2,final_range_RK2,z_new_RK2[RK2_neg_index-1]


RK2_x=np.loadtxt('RK2xcoordinates.dat')
RK2_y=np.loadtxt('RK2ycoordinates.dat')
RK2_z=np.loadtxt('RK2zcoordinates.dat')
final_range_RK2=np.amax(RK2_x)
final_height_RK2=np.amax(RK2_y)
max_z_deflection=np.amax(RK2_z)
max_z_deflection=np.amin(RK2_z)


if (DO==5):
#### In absence of air resistance and other influencing factors
   v_0=45                   # initial velocity in ms^-1 (100.662 mph)
   angle=40                 # angle in degrees   
   angle_rad=(pi*angle)/180 # initial angle (at t=0) in radians
   vx_0=v_0*cos(angle_rad)  # initial velocity (at t=0) in x_direction
   vy_0=v_0*sin(angle_rad)  # initial velocity (at t=0) in y_direction
   b_m=0.149                # Mass of baseball in kg
   B1=0.0                   # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.00                   # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225              # Density at sea level (approximate)
   A=4.5365E-3              # Frontal area (Baseball frontal area in m^2)
   d=0.076                  # diameter of baseball
   B2_m=8.76465E-3          # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4)
   h1=0.00041               # step size (timestep) in s
   h=h1
   N=500000;                # number of euler steps (number of timesteps)
   T=273                    # Temperature in Kelvin
   Kb=1.38064852E-23        # Boltzman constant
   S0=0.06109               # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=0.0                 # Not considering spin
   rpm=0.0                  # revolutions per second
   omega=2*pi*rpm           # radians per second
   final_height_RK2_without_resistance,final_range_RK2_without_resistance,x_new_RK2_WO_R,y_new_RK2_WO_R,z_new_RK2_WO_R,vx_new_RK2_WO_R,vy_new_RK2_WO_R,vz_new_RK2_WO_R,RK2_neg_index_WO_R=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)

   x_new_RK2_WO_R=x_new_RK2_WO_R[:RK2_neg_index_WO_R]
   y_new_RK2_WO_R=y_new_RK2_WO_R[:RK2_neg_index_WO_R]
   z_new_RK2_WO_R=z_new_RK2_WO_R[:RK2_neg_index_WO_R]

   np.savetxt('RK2NRxcoordinates.dat', x_new_RK2_WO_R)
   np.savetxt('RK2NRycoordinates.dat', y_new_RK2_WO_R)
   np.savetxt('RK2NRzcoordinates.dat', z_new_RK2_WO_R)
   np.savetxt('RK2NRvxcoordinates.dat', vx_new_RK2_WO_R)
   np.savetxt('RK2NRvycoordinates.dat', vy_new_RK2_WO_R)
   np.savetxt('RK2NRvzcoordinates.dat', vy_new_RK2_WO_R)
   print "Horizontal range when there is no resistance, using RK 2nd order="
   print final_range_RK2_without_resistance
   print "Vertical range when there is no resistance, using RK 2nd order="
   print final_height_RK2_without_resistance


   h2=0.00041
   h=h2
   final_height_Euler_without_resistance,final_range_Euler_without_resistance,x_new_Euler_WO_R,y_new_Euler_WO_R,z_new_Euler_WO_R,vx_new_Euler_WO_R,vy_new_Euler_WO_R,vz_new_Euler_WO_R,Euler_neg_index_WO_R=euler_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)

   x_new_Euler_WO_R=x_new_Euler_WO_R[:Euler_neg_index_WO_R]
   y_new_Euler_WO_R=y_new_Euler_WO_R[:Euler_neg_index_WO_R]
   z_new_Euler_WO_R=z_new_Euler_WO_R[:Euler_neg_index_WO_R]

   np.savetxt('EulerNRxcoordinates.dat', x_new_Euler_WO_R)
   np.savetxt('EulerNRycoordinates.dat', y_new_Euler_WO_R)
   np.savetxt('EulerNRzcoordinates.dat', z_new_Euler_WO_R)
   np.savetxt('EulerNRvxcoordinates.dat', vx_new_Euler_WO_R)
   np.savetxt('EulerNRvycoordinates.dat', vy_new_Euler_WO_R)
   np.savetxt('EulerNRvzcoordinates.dat', vy_new_Euler_WO_R)
   print "Horizontal range when there is no resistance, using euler="
   print final_range_Euler_without_resistance
   print "Vertical range when there is no resistance, using euler="
   print final_height_Euler_without_resistance

   euler_NR_x=np.loadtxt('EulerNRxcoordinates.dat')
   euler_NR_y=np.loadtxt('EulerNRycoordinates.dat')
   euler_NR_z=np.loadtxt('EulerNRzcoordinates.dat')

   RK2_NR_x=np.loadtxt('RK2NRxcoordinates.dat')
   RK2_NR_y=np.loadtxt('RK2NRycoordinates.dat')
   RK2_NR_z=np.loadtxt('RK2NRzcoordinates.dat')

#####   NO AIR RESISTANCE PLOT BLOCK (UNCOMMENT TO USE) 
   #fig, ax = plt.subplots()
###     Uncomment following two lines for XZ plot 
   #ax.plot(euler_NR_x,euler_NR_z,'r-x',label='XZ-Plane Euler Forward, h=0.000041')
   #ax.plot(RK2_NR_x,RK2_NR_z,'y--',label='XZ-Plane Runge-Kutta 2nd Order, h=0.000041')
###     Uncomment following two lines for XY plot
   #ax.plot(euler_NR_x,euler_NR_y,'r-x',label='XY-Plane Euler Forward, h=0.000041')
   #ax.plot(RK2_NR_x,RK2_NR_y,'y--',label='XY-Plane Runge-Kutta 2nd Order, h=0.000041')
   #ax.set_title('PROJECTILE (NO RESISTANCE) XY PLANE')
   #ax.grid(True)
   #xlabel('X Range (m)')
   #ylabel('Height (m)')
   #ylabel('Distance along Z-axis (m)')
   #legend(loc='best')

   #savefig('NO_R_XYProjectile_compare.png')
####   NO AIR RESISTANCE 3D PLOT BLOCK (UNCOMMENT TO USE)
   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(euler_NR_x, euler_NR_y, euler_NR_z, 'r-x', label='No resistance trajectory Euler, h=0.000041')
   ax.plot(RK2_NR_x, RK2_NR_y, RK2_NR_z,'y--', label='No resistance trajectory 2nd Order RK, h=0.000041')
   ax.set_xlabel('X (Horizonatal Range (m))')
   ax.set_ylabel('Y (Elevation from ground (m))')
   ax.set_zlabel('Z (distance along magnus force (m))')
   legend(loc='best')
   ax.view_init(azim=10)
   savefig('No_R_3DProjectile.png')

   initial_height=0.0  #For analytical horizontal range R in the absence of air resistance
   Analytic_rangeR=analytical_rangeR(v_0, angle_rad, initial_height)
   print "Analytically found horizontal range when there is no resistance="
   print Analytic_rangeR

if (DO==6):
######## DIFFERENT ANGLES  
   v_0=45                 # initial velocity in ms^-1 
   b_m=0.149              # Mass of baseball in kg
   B1=0.0                 # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                 # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225            # Density at sea level (approximate)
   A=4.5365E-3            # Frontal area (Baseball frontal area in m^2)
   d=0.076                # diameter of baseball
   B2_m=8.76465E-3        # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4). I dont use this when effect of altitude is considered
   h2=0.000041            # step size (timestep) in s
   h=h2
   N=500000;              # number of euler steps (number of timesteps)
   T=273                  # Temperature in Kelvin
   Kb=1.38064852E-23      # Boltzman constant
   S0=0.06109             # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=4.1E-4            # Taken from textbook (G&N) for baseball.
   rpm=30                 # revolutions per second
   omega=2*pi*rpm         # radians per second

   Num_angle_iter=25
   angles_degree=np.linspace(20,70,Num_angle_iter)
   angles_radians=(pi*angles_degree)/180

   final_height_RK2=np.zeros(Num_angle_iter)
   final_range_RK2=np.zeros(Num_angle_iter)
   all_xranges_RK2=np.zeros(Num_angle_iter)
   neg_index=np.zeros(Num_angle_iter)
   for i in range(0,Num_angle_iter,1):
     vx_0=v_0*cos(angles_radians[i])
     vy_0=v_0*sin(angles_radians[i])
     final_height_RK2[i],final_range_RK2[i],x_new,y_new,z_new,vx_new,vy_new,vz_new,neg_index[i]=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)
     all_xranges_RK2[i]=final_range_RK2[i]
     temp=int(neg_index[i])
#     print temp
     x_new=x_new[:temp]
     y_new=y_new[:temp]
     z_new=z_new[:temp]
#####   ANGLE DEPENDENCE IN PRESENCE OF OTHER EFFECTS + AIR RESISTANCE PLOT BLOCK (UNCOMMENT TO USE) 
     plt.plot(x_new, y_new, label=angles_degree[i]) 
   #  plt.plot(x_new, z_new, label=angles_degree[i]) 
   #  plt.axis([0, 50, 5, -10])
     plt.axis([0, 500, 0, 150])
     plt.xlabel('X Range (m)')
     plt.ylabel('Height (m)')
   #  plt.ylabel('Distance moved along Z (m)')
   #  plt.grid(True)
     plt.title('PROJECTILE WITH A GIVEN VELOCITY AT DIFFERENT\n ANGLES (DEGREES) XY PLANE (RK2)')
     legend(loc='best',prop={'size': 8})
     savefig('ALL_EFFECTS_XYRK_Angles_Air_Altitude_Spin_projectile.png')

   for i in range(1,Num_angle_iter,1):
     index_max_x=argmax(all_xranges_RK2)
     Opt_angle= angles_degree[index_max_x]
   print "Optimal initial angle (degree) is=",Opt_angle,"for initial velocity (m/s) =", v_0, "\n to get max x range =", all_xranges_RK2[index_max_x]

if (DO==7):
########EFFECT OF DIFFERENT ANGLES IN ABSENCE OF ANY INFLUENCING FACTOR  
   v_0=45                 # initial velocity in ms^-1 
   b_m=0.149              # Mass of baseball in kg
   B1=0.0                 # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.00                 # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225            # Density at sea level (approximate)
   A=4.5365E-3            # Frontal area (Baseball frontal area in m^2)
   d=0.076                # diameter of baseball
   B2_m=8.76465E-3        # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4). I dont use this when effect of altitude is considered
   h2=0.000041            # step size (timestep) in s
   h=h2
   N=500000;              # number of euler steps (number of timesteps)
   T=273                  # Temperature in Kelvin
   Kb=1.38064852E-23      # Boltzman constant
   S0=0.06109             # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=0.0
   rpm=0.0                # revolutions per second
   omega=2*pi*rpm         # radians per second

   Num_angle_iter=25
   angles_degree=np.linspace(20,70,Num_angle_iter)
   angles_radians=(pi*angles_degree)/180

   final_height_RK2=np.zeros(Num_angle_iter)
   final_range_RK2=np.zeros(Num_angle_iter)
   all_xranges_RK2=np.zeros(Num_angle_iter)
   neg_index=np.zeros(Num_angle_iter)
   for i in range(0,Num_angle_iter,1):
     vx_0=v_0*cos(angles_radians[i])
     vy_0=v_0*sin(angles_radians[i])
     final_height_RK2[i],final_range_RK2[i],x_new,y_new,z_new,vx_new,vy_new,vz_new,neg_index[i]=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)
     all_xranges_RK2[i]=final_range_RK2[i]
     temp=int(neg_index[i])
#     print temp
     x_new=x_new[:temp]
     y_new=y_new[:temp]
     z_new=z_new[:temp]
#####   ANGLE DEPENDENCE IN ABSENCE OF OTHER EFFECTS + AIR RESISTANCE PLOT BLOCK (UNCOMMENT TO USE) 
     plt.plot(x_new, y_new, label=angles_degree[i]) 
   #  plt.plot(x_new, z_new, label=angles_degree[i]) 
   #  plt.axis([0, 50, 5, -10])
     plt.axis([0, 500, 0, 150])
     plt.xlabel('X Range (m)')
     plt.ylabel('Height (m)')
   #  plt.ylabel('Distance moved along Z (m)')
   #  plt.grid(True)
     plt.title('PROJECTILE WITH A GIVEN VELOCITY AT DIFFERENT ANGLES\n (DEGREES) IN ABSENCE OF OTHER FACTORS, XY PLANE (RK2)')
     legend(loc='best',prop={'size': 8})
     savefig('XYRK_Angles_Absence_projectile.png')

   for i in range(1,Num_angle_iter,1):
     index_max_x=argmax(all_xranges_RK2)
     Opt_angle= angles_degree[index_max_x]
   print "Optimal initial angle (degree) is=",Opt_angle,"for initial velocity (m/s) =", v_0, "\n to get max x range =", all_xranges_RK2[index_max_x]

vel_comps=np.loadtxt('final_vel_components.dat')
prev_vx_RK=vel_comps[0]
prev_vy_RK=vel_comps[1]
prev_vz_RK=vel_comps[2]

########Finding initial conditions by traversing back to initial conditions from final conditions. Reverse path.
if (DO==8):
   b_m=0.149              # Mass of baseball in kg
   B1=0.0                 # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                 # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225            # Density at sea level (approximate)
   A=4.5365E-3            # Frontal area (Baseball frontal area in m^2)
   d=0.076                # diameter of baseball
   B2_m=8.76465E-3        # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4). I dont use this when effect of altitude is considered
   h2=0.000041            # step size (timestep) in s
   h=h2
   N=500000;              # number of euler steps (number of timesteps)
   T=273                  # Temperature in Kelvin
   Kb=1.38064852E-23      # Boltzman constant
   S0=0.06109             # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=4.1E-4            # Taken from textbook (G&N) for baseball.
   rpm=30                 # revolutions per second
   omega=2*pi*rpm         # radians per second

   print "Final velocity from task 4 (2RK) is used here"
   v_fi=sqrt(pow(prev_vx_RK,2)+pow(prev_vy_RK,2)+pow(prev_vz_RK,2))
   print "Using previous maximum Horizontal range and maximum z deflection. Final angle is calculated using the Final velocity from task 4 (2RK).\n This is done to show that this function successfully travels from final position to initial position and retriews initial conditions that were input in task 4.\n So run task 4 first and then run this task"
   HR=final_range_RK2
#   HR=input("Enter the horizontal range=") 
   ZR=max_z_deflection
#   ZR=input("Enter the range in Z direction (negative value because of magnus force acting in negative direction)=")
   fi_alpha_ang_rad=acos((prev_vx_RK/v_fi))
#   fi_alpha_ang_degree=input("Enter the final angle alpha (angle with +x-axis)=(43 for the above range)")
#   fi_alpha_ang_rad=(pi*fi_alpha_ang_degree)/180
   fi_beta_ang_rad=acos((prev_vy_RK/v_fi))
#   fi_beta_ang_degree=input("Enter the final angle beta (angle with +y-axis (90<beta<180))=(119 for the above range)")
#   fi_beta_ang_rad=(pi*fi_beta_ang_degree)/180
   fi_gamma_ang_rad=acos((prev_vz_RK/v_fi))
#   fi_gamma_ang_degree=input("Enter the final angle beta (angle with +z-axis (90<beta<180))=(119 for the above range)")
#   fi_gamma_ang_rad=(pi*fi_gamma_ang_degree)/180
   vx_0=v_fi*cos(fi_alpha_ang_rad)
   vy_0=v_fi*cos(fi_beta_ang_rad)
   vz_0=v_fi*cos(fi_gamma_ang_rad)
#   v_fi=input("Enter Final velocity=")
   print fi_alpha_ang_rad, fi_beta_ang_rad, fi_gamma_ang_rad
   Max_reverse_height_euler,final_reverse_range_euler,x_new_reverse_euler,y_new_reverse_euler,z_new_reverse_euler,vx_new_reverse_euler,vy_new_reverse_euler,vz_new_reverse_euler,euler_neg_index=euler_reverse_trajectory(omega, S0, Kb, T, C, A, v_fi, vx_0, vy_0, vz_0, b_m, h, N, rho_0, S0_m)
   x_new_reverse_euler=x_new_reverse_euler[euler_neg_index:]
   y_new_reverse_euler=y_new_reverse_euler[euler_neg_index:]
   z_new_reverse_euler=z_new_reverse_euler[euler_neg_index:]

   vx_new_reverse_euler=vx_new_reverse_euler[euler_neg_index:]
   vy_new_reverse_euler=vy_new_reverse_euler[euler_neg_index:]
   vz_new_reverse_euler=vz_new_reverse_euler[euler_neg_index:]

   np.savetxt('x_reversecoordinates.dat', x_new_reverse_euler)
   np.savetxt('y_reversecoordinates.dat', y_new_reverse_euler)
   np.savetxt('z_reversecoordinates.dat', z_new_reverse_euler)
   np.savetxt('vx_reversecoordinates.dat', vx_new_reverse_euler)
   np.savetxt('vy_reversecoordinates.dat', vy_new_reverse_euler)
   np.savetxt('vz_reversecoordinates.dat', vy_new_reverse_euler)

   print "Initial velocity in m/s (is it 45 m/s? If it is, then initial velocity has been successfully found!)="
   v0_ini=sqrt(pow(vx_new_reverse_euler[0],2)+pow(vy_new_reverse_euler[0],2)+pow(vz_new_reverse_euler[0],2))
   print v0_ini

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   ax.plot(x_new_reverse_euler, y_new_reverse_euler, z_new_reverse_euler, label='Reverse Ball trajectory curve')
   ax.set_xlabel('X (Horizonatal Range (m))')
   ax.set_ylabel('Y (Elevation from ground (m))')
   ax.set_zlabel('Z (distance along magnus force (m))')
   legend(loc='best')
   ax.view_init(azim=10)
   savefig('Reverse3DProjectile.png')


v_0=45                          # task 1 initial velocity in ms^-1 (100.662 mph)
angle=40                        # task 1 angle in degrees   
angle_rad=(pi*angle)/180        # task 1 initial angle (at t=0) in radians
print "Initial velocity used in task 4=",v_0,"and initial angle used in task 4=",angle_rad,"radians,",angle,"degrees"

########## Iterating over guessed initial conditions to get final conditions
if (DO==9):
   ######FINDING INITIAL CONDITIONS 
   Num_Iter=200      #200 for v_0=20 m/s and angle=80 degree. Better the guess (velocity or angle), lower the value of Num_Iter
   final_height_NRK2=np.zeros(Num_Iter)
   final_range_NRK2=np.zeros(Num_Iter)
   b_m=0.149              # Mass of baseball in kg
   B1=0.0                 # Stoke's drag which is neglected and set to zero for macroscopic objects
   C=0.47                 # Shape dependent constant (experimentally determined C-values) for sphere (C=1 for with air drag, C=0 for no air drag)
   rho_0=1.225            # Density at sea level (approximate)
   A=4.5365E-3            # Frontal area (Baseball frontal area in m^2)
   d=0.076                # diameter of baseball
   B2_m=8.76465E-3        # Calculated by ((0.5*C*rho_0*A)/b_m) where A=((pi*d*d)/4). I dont use this when effect of altitude is considered
   h2=0.000041            # step size (timestep) in s
   h=h2
   N=500000;              # number of euler steps (number of timesteps)
   T=273                  # Temperature in Kelvin
   Kb=1.38064852E-23      # Boltzman constant
   S0=0.06109             # Taken from textbook (G&N) for baseball. This value here is per kg.  
   S0_m=4.1E-4            # Taken from textbook (G&N) for baseball.
   rpm=30                 # revolutions per second
   omega=2*pi*rpm         # radians per second


   print "Using previous maximum Horizontal and vertical ranges from task 4. Using these ranges, the initial angle and velocty of task 4 can be retriewed here. \nSo run task 4 first and then run this task"

   GHR=final_range_RK2     #input("Enter the horizontal range")
   GVR=final_height_RK2    #input("Enter the vertical range")

   task_id=input("Enter 1 if there is need for angle of the initial velocity (given) with given horizontal and vertical ranges or enter 2 if there is need for velocity with given angle and ranges\n")
   print "In task ",DO," Option=",task_id," is chosen\n"
   angles2_degree=np.linspace(38.5, 41.5, Num_Iter) # for v_0=45m/s and angle=40degrees-->np.linspace(38.5, 41.5, Num_Iter)
   angles2_radians=(pi*angles2_degree)/180
   range_angles=[]
   range_vels=[]

   neg1_index=np.zeros(Num_Iter)
   error1=np.zeros(Num_Iter)
   error2=np.zeros(Num_Iter)
   v_0new=np.zeros(Num_Iter)
   neg2_index=np.zeros(Num_Iter)
   z_defl=np.zeros(Num_Iter)
   if (task_id==1):
    print "Initial velocity from task 4 (2RK) is used here and different initial angles are tried to get required final ranges"
    v_0=v_0
    #v_0=input("Enter the velocity in m/s")
    for i in range(0,Num_Iter,1):
        vx_0=v_0*cos(angles2_radians[i])
        vy_0=v_0*sin(angles2_radians[i])
        final_height_NRK2[i],final_range_NRK2[i],x_new,y_new,z_new,vx_new,vy_new,vz_new,neg1_index=RK2_trajectory(omega, S0, Kb, T, C, A, v_0, vx_0, vy_0, b_m, h, N, rho_0, S0_m)
        z_new=z_new[:neg1_index]
        z_defl[i]=np.amin(z_new)
#        print angles2_degree[i],z_defl[i]
        #z_new=z_new[:neg1_index]
        if ((-0.05<=(final_range_NRK2[i]-GHR)<= 0.05) and ((-0.05<=(final_height_NRK2[i]-GVR)<= 0.05))):  ##condition checking the calculated final ranges are close enough to given final ranges
          range_angles.append(i)
          error1[i]=final_range_NRK2[i]-GHR
          error2[i]=final_height_NRK2[i]-GVR
          least_error1_index=argmin(error1)
          close_angle=angles2_degree[least_error1_index]
          print "Guessed initial angle in degree=",angles2_degree[i],"gives final horizontal range(m)=",final_range_NRK2[i],"final vertical range(m)=",final_height_NRK2[i],"which are comparable to", GHR, GVR, "respectively","final z-axis deflection(m)=",z_defl[i]
    print "Therefore I choose initial angle=", close_angle,"in degree"
   else:
    v_0=np.linspace(44.5, 45.5, Num_Iter)  # very wide-range and a lot of iterations required if there is no clue of what the initial velocity could be
    ini_angle=angle
#    ini_angle=input("Enter the initial angle in degree")
    ini_angle_rad=(pi*ini_angle)/180
    for i in range(0,Num_Iter,1):
        vx_0=v_0[i]*cos(ini_angle_rad)
        vy_0=v_0[i]*sin(ini_angle_rad)
        final_height_NRK2[i],final_range_NRK2[i],x_new,y_new,z_new,vx_new,vy_new,vz_new,neg1_index=RK2_trajectory(omega, S0, Kb, T, C, A, v_0[i], vx_0, vy_0, b_m, h, N, rho_0, S0_m)
        z_new=z_new[:neg1_index]
        z_defl[i]=np.amin(z_new)
        #print i,final_height_NRK2[i],final_range_NRK2[i]
        #z_new=z_new[:neg1_index]
        if ((-0.05<=(final_range_NRK2[i]-GHR)<= 0.05) and ((-0.05<=(final_height_NRK2[i]-GVR)<= 0.05))):
          range_vels.append(i)
          error1[i]=final_range_NRK2[i]-GHR
          error2[i]=final_height_NRK2[i]-GVR
          least_error1_index=argmin(error1)
          close_vel=v_0[least_error1_index]
          print "Guessed initial velocity=",v_0[i],"gives final horizontal range(m)=",final_range_NRK2[i],"final vertical range(m)=",final_height_NRK2[i],"which are comparable to", GHR, GVR, "respectively","final z-axis deflection(m)=",z_defl[i]
    print "Therefore I choose initial velocity=", close_vel, "in m/s"

