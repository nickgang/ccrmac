"""
- mdct.py -- Computes reasonably fast MDCT/IMDCT using numpy FFT/IFFT
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
# import mdct_ as m
import time

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """
    ### YOUR CODE STARTS HERE ###
    n0=(b+1)*0.5 # Transform kernel, can be variable for block switching

    if isInverse==False:
        if type(data)!=np.ndarray:
            data = np.array(data) # make sure to pass numpy array, not python array

        N = data.size
        nVec = np.arange(0,N)# vector of indeces
        X = np.zeros(a)
        for k in range(a): # Compute transform value for each bin
            X[k] = (2/float(N))*np.dot(data,np.cos(((2*np.pi)/float(N))*(nVec+n0)*(k + 0.5)))

        return X

    elif isInverse==True:
        N = data.size*2
        kVec = np.arange(0,N)
        X = np.zeros(N) # Make room for N frequency bins
        X[0:(N/2.0)] = data
        X[N/2.0:N] = -1*np.fliplr([data])[0] # add in second half of spectrum

        x = np.zeros(N)
        for n in range(N):
            x[n] = np.dot(X,np.cos(((2*np.pi)/float(N))*(n+n0)*(kVec+0.5))) # Leaving out factor of 2/N

        return x

    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    (and where 2/N factor is included in forward transform instead of inverse)
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    if isInverse==False:
        if type(data)!=np.ndarray:
            data = np.array(data) # make sure numpy array not python array

        N = data.size
        n0 = (b+1)*0.5 # Transform kernel, can be variable for block switching
        nVec = np.arange(0,N) # vector of indeces
        kVec = np.arange(0,N/2) # vector of bins (0 to N/2-1)

        preTwiddle = data*np.exp((-1j*2*np.pi*nVec)/float(2*N)) # 1. pre-twiddle
        X = np.fft.fft(preTwiddle,N) # 2. transform pre-twiddled data
        # 3. post-twiddle the real part of first half of the bins
        postTwiddle = (2/float(N))*(np.exp(((-1j*2*np.pi*n0)/float(N))*(kVec+0.5))*X[kVec]).real

    elif isInverse==True:
        N = data.size*2
        n0 = (b+1)*0.5 # Transform kernel, can be variable for block switching
        nVec = np.arange(0,N) # vector of indeces
        kVec = np.arange(0,N) # vector of indeces

        X = np.zeros(N) # Make room for N frequency bins
        X[0:(N/2)] = data
        X[N/2:N] = -1*np.fliplr([data])[0] # add in second half of spectrum

        preTwiddle = np.exp(((1j*2*np.pi*n0*kVec)/float(N)))*X # 1. pre-twiddle N-point spectrum
        x = np.fft.ifft(preTwiddle,N) # 2. Transform pre-twiddled data
        postTwiddle = float(N)*(np.exp(((1j*np.pi)/float(N))*(nVec+n0))*x).real # 3. post-twiddle

    return postTwiddle
    ### YOUR CODE ENDS HERE ###

def IMDCT(data,a,b):
    ### YOUR CODE STARTS HERE ###
    return MDCT(data,a,b,True)
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------
