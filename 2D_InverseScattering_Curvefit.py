#!/usr/bin/env python
# coding: utf-8

# In[23]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.optimize import curve_fit


# In[24]:


def PointsInCircum(r,n):    #function that finds the cartesian coordinates of n evenly distributed points along the 
    Plist = []              #circumference of a circle of radius r
    
    for i in range(0,n):
    
        Plist.append((np.cos(2*np.pi/n*i)*r, np.sin(2*np.pi/n*i)*r))
    
    return np.array(Plist)


# In[25]:


def Incident(r):    #function calculating the incident field at a given point r

    E0 = E_0 * np.exp(1j * k_0 * r[0])
    
    return E0    


# In[26]:


def Field(R,P):    #function calculating the total field by summing the incident and scatterer fields (ignoring multiple
                   #scattering events)
    EjCont = 0                    
    
    for i in range(0,len(P)):
        EjCont += sp.special.hankel1(0,k_0 * np.linalg.norm(np.subtract(R,P[i]))) * Incident(P[i])
    
    return Incident(R) + EjCont    

Field([0,0], P)


# In[27]:


def v1Field(data,Px,Py):    #fucntions used to find the field for an array of points with up to 6 sactterers
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py]]))
    return np.real(np.array(field))

def v2Field(data,Px,Py,P1x,P1y):
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py],[P1x,P1y]]))
    return np.real(np.array(field))

def v3Field(data,Px,Py,P1x,P1y,P2x,P2y):
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py],[P1x,P1y],[P2x,P2y]]))
    return np.real(np.array(field))

def v4Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y):
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py],[P1x,P1y],[P2x,P2y],[P3x,P3y]]))
    return np.real(np.array(field))

def v5Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y):
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py],[P1x,P1y],[P2x,P2y],[P3x,P3y],[P4x,P4y]]))
    return np.real(np.array(field))

def v6Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y,P5x,P5y):
    field = []
    for i in range(0,len(data)):
        field.append(Field(data[i],[[Px,Py],[P1x,P1y],[P2x,P2y],[P3x,P3y],[P4x,P4y],[P5x,P5y]]))
    return np.real(np.array(field))


# In[28]:


def X(params,P,N):    #funtion that calculates the chi squared value for the results produced by the curve fit

    for i in range(0,N):
        for j in range(0,1):

            x = 0
            x += ((params[i+j] - P[i][j])**2)/P[i][j]
    return x


# In[29]:


def find_scatter(N,P):    #function that finds the best fit for the field and locates the position of each scatterer  
    
    r = np.max(P) + 5
    data = PointsInCircum(r,1000)
    
    while True:           #first loop writes the coordinates as individual variables and uses these as inputs for the      
        Px =  P[0,0]      #functions previously defined
        Py =  P[0,1]
        Z = v1Field(data,Px,Py)
    
        if N == 1:
            break         #loop breaks so it only calculates the field for the correct number of scatterers
        else:
            P1x = P[1,0]
            P1y = P[1,1]
            Z = v2Field(data,Px,Py,P1x,P1y)

            if N == 2:
                break
            else:
                P2x = P[2,0]
                P2y = P[2,1]
                Z = v3Field(data,Px,Py,P1x,P1y,P2x,P2y)

                if N == 3:
                    break
                else:
                    P3x = P[3,0]
                    P3y = P[3,1]
                    Z = v4Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y)

                    if N == 4:
                        break
                    else:
                        P4x = P[4,0]
                        P4y = P[4,1]

                        Z = v5Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y)

                        if N == 5:
                            break
                        else:
                            P5x = P[5,0] 
                            P5y = P[5,1]

                            break
    
    
    
    x = 1000
    while x > 0.00001:    #second loop applies a curve fit. It applies the data using each function until it finds a function
                          #that gives a result with a low enough standard deviation for it to be correct
        params, pcov = sp.optimize.curve_fit(v1Field, data, Z, maxfev=5000)
        print('1 Scatterer:')
        print(params)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('Average standard deviation:', x)

        if x < 0.0001:    #the correct result will have sd on scale of e-16 and incorrect fits can be on the scale of 0.01-10000
            break    #the loop breaks when a result with low enough uncertainty is found
        else:

            params, pcov = sp.optimize.curve_fit(v2Field, data, Z, maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            print('2 Scatterers:')
            print(params)
            x = np.mean(perr)
            print('Average standard deviation:',x)

            if x < 0.0001:
                break  
            else:

                params, pcov = sp.optimize.curve_fit(v3Field, data, Z, maxfev=5000)
                perr = np.sqrt(np.diag(pcov))
                print('3 Scatterers:')
                print(params)
                x = np.mean(perr)
                print('Average standard deviation:',x)

                if x < 0.0001:
                    break
                else:

                    params, pcov = sp.optimize.curve_fit(v4Field, data, Z, maxfev=5000)
                    perr = np.sqrt(np.diag(pcov))
                    print('4 Scatterers:')
                    print(params)
                    x = np.mean(perr)
                    print('Average standard deviation:',x)

                    if x < 0.0001:
                        break
                    else:

                        params, pcov = sp.optimize.curve_fit(v5Field, data, Z, maxfev=5000)
                        print('5 Scatterers:')
                        print(params)
                        perr = np.sqrt(np.diag(pcov))
                        x = np.mean(perr)
                        print('Average standard deviation:',x)

                        if x < 0.0001:
                            break
                        else:

                            params, pcov = sp.optimize.curve_fit(v6Field, data, Z, maxfev=5000)
                            print('6 Scatterers:')
                            print(params)
                            perr = np.sqrt(np.diag(pcov))
                            x = np.mean(perr)
                            print('Average standard deviation:',x)


# In[36]:


def find_scatter_all(N,P):
    
          
        """This function locates the positions of scatterers by curve fitting a scattered, 2D, field. Due to the rigid
           nature of python functions a variable number of inputs cannot be used. Due to this the code is limited to only
           6 scatterers. Function itself generates the field and ten fits the curve to it, so the standard deviation for
           each result can be found. This function creates, and prints, a fit for N = 1 to = N to show how the standard
           deviation varies.
           
           parameters:
               
               N (integer): Number of scatterers
               
               P (np.array): Position(s) of scatterers (e.g [[1,2],[3,4],[5,6]])
               
            prints:
            
               for i in N:
                   
                   i (number of scatterers)
                   scatterer location(s) (e.g [[1,1],[3,5]])
                   standard deviation
            
            returns:
            
                None
        """
        
        r = np.max(P) + 5
        data = PointsInCircum(r,1000)

        while True:
            Px =  P[0,0] 
            Py =  P[0,1]
            Z = v1Field(data,Px,Py)

            if N == 1:
                break
            else:
                P1x = P[1,0]
                P1y = P[1,1]
                Z = v2Field(data,Px,Py,P1x,P1y)

                if N == 2:
                    break
                else:
                    P2x = P[2,0]
                    P2y = P[2,1]
                    Z = v3Field(data,Px,Py,P1x,P1y,P2x,P2y)

                    if N == 3:
                        break
                    else:
                        P3x = P[3,0]
                        P3y = P[3,1]
                        Z = v4Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y)

                        if N == 4:
                            break
                        else:
                            P4x = P[4,0]
                            P4y = P[4,1]

                            Z = v5Field(data,Px,Py,P1x,P1y,P2x,P2y,P3x,P3y,P4x,P4y)

                            if N == 5:
                                break
                            else:
                                P5x = P[5,0] 
                                P5y = P[5,1]

                                break

        params, pcov = sp.optimize.curve_fit(v1Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('1 Scatterer:')
        print(params)
        print('chi2 =',X(params,P,2))    #this function also prints the chi squared value for each result
        print('std=',x)
        print(' ')

        params, pcov = sp.optimize.curve_fit(v2Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('2 Scatterers:')
        print(params)
        #print('chi2 =',X(params,P,4))    #if the number of results produced is larger than the actual number of scatterers then
        print('std=',x)                   #chi squared cannot be calculated
        print(' ')

        params, pcov = sp.optimize.curve_fit(v3Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('3 Scatterers:')
        print(params)
        #print('chi2 =',X(params,P,6))
        print('std=',x)
        print(' ')

        params, pcov = sp.optimize.curve_fit(v4Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('4 Scatterers:')
        print(params)
        #print('chi2 =',X(params,P,8))
        print('std=',x)
        print(' ')

        params, pcov = sp.optimize.curve_fit(v5Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('5 Scatterers:')
        print(params)
        #print('chi2 =',X(params,P,10))
        print('std=',x)
        print(' ')

        params, pcov = sp.optimize.curve_fit(v6Field, data, Z, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x = np.mean(perr)
        print('6 Scatterers:')
        print(params)
        print('chi2 =',X(params,P,12))
        print('std=',x)


# In[38]:


k_0 = 0.2    
E_0 = 20
N = np.random.randint(1,3)    #number of scatterers
P = np.array(np.random.randint(1,5,(N,2)))    #location of scatterers
#P = np.array([[1,1]])    #manual selection of location of scatterers
#N = 1    #manual selection of number of scatterers
print(P)
print(' ')

find_scatter(N,P)

