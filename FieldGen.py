#!/usr/bin/env python
# coding: utf-8

# In[78]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import curve_fit


# In[80]:


def Incident(r):

    E0 = E_0 * np.exp(1j * k_0 * r[0])
    
    return E0    #Calculating the incident field at a given point r#

vIncident = np.vectorize(Incident, signature='(n)->()') 
#vIncident = np.vectorize(Incident, excluded=['E_0,k_0'])


# In[81]:


def Field(R,e_0,K_0,P):
    
    N = len(P)
    k_0 = K_0    #Defining all required constants#
    E_0 = e_0
    
    EjCont = 0                           #Setting the initial value of the scattered field contribution#
    M = np.zeros((N,N),dtype=complex)    #Creating the matrix and forcing values to be complex#

    for i in range(0,N):
        for j in range(0,N):            
            if i == j:
                M[i][j] = 1
            else:
                M[i][j] = sp.special.hankel1(0,k_0 * np.linalg.norm(np.subtract(P[i],P[j])))    
          
    Ej = np.linalg.solve(M,vIncident(P))    #Solving the matrix equation to give the field at each point#
    
    for i in range(0,N):
        EjCont += sp.special.hankel1(0,k_0 * np.linalg.norm(np.subtract(R,P[i]))) * Ej[i]
    
    return Incident(R) + EjCont    #Calculating the total field by summing the incident and scatterer fields#


# In[82]:


def Field_Generator(size,step,P,E_0,k_0):
    
    """This function generates the field created by the scatterering of an incident wave by n point like
       scatterers. It can generate fields for any number of scatterers and, if ignoring computational costs,
       of unlimate size.
       
       Parameters:
       
       size (float): The width of the field generate (size > 0)
       
       step (float): The distance between each recorded field point
       
       P (np.array): An array containing the coordiantes of the scatterers e.g [[a,b]] (a,b > 0)
       
       E_0 (float): Incident wave amplitude
       
       k_0 (float): Incident wave number
       
       Returns:
       
       X (np.array): An array containg x coordinates for every recorded point
       
       Y (np.array): An array containg y coordinates for every recorded point
       
       Z (np.array): A 2D array containg the field value at every point
       """
    
    xs = np.arange(0,size+step,step)
    ys = np.arange(0,size+step,step)
    
    X,Y = np.meshgrid(xs,ys)
    
    Z = np.zeros((len(X),len(X)), dtype=complex)
    
    for i in range(0,len(X)):
        for j in range(0,len(Y)):
            Z[i][j] = Field([X[i][j],Y[i][j]],E_0,k_0,P)
            
    return X,Y,Z


