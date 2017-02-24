"""
window.py -- Defines functions to window an array of data samples
Written by Nick Gang for Music 422, 1/31/17
"""

import numpy as np
# import window_ as w
import matplotlib.pyplot as plt

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = dataSampleArray.size
    nVec = np.arange(0,N)

    sWin = np.sin((np.pi*(nVec+0.5))/float(N))
    output = sWin*dataSampleArray

    return  output
    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = dataSampleArray.size
    nVec = np.arange(0,N)

    hWin = 0.5*(1-np.cos((2*np.pi*(nVec+0.5))/float(N)))
    output = hWin*dataSampleArray

    return  output
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following pp. 108-109 and pp. 117-118 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """
    ### YOUR CODE STARTS HERE ###
    n = len(dataSampleArray)
    kb = np.arange(n/2+1) # from 0 to n/2 (n/2+1 values)
    kb = np.i0(np.pi*alpha*np.sqrt(1.0 - (4.0*kb/n - 1.0)**2))/np.i0(np.pi*alpha)
    # normalize KB window into a KBD window (see book p. 117, but note there's
    # a typo in the book, where the power of 2 was left out by mistake)
    d = np.zeros(n) # allocate memory
    denom = sum(kb**2) # denominator to normalize running sum
    d[:n/2] = np.cumsum(kb[:-1]**2)/denom # 1st half is normalized running sum
    d[n/2:] = d[:n/2][::-1] # 2nd half is just the reverse of 1st half
    d = np.sqrt(d) # take square root of elements
    # window samples and return
    d *= dataSampleArray
    return np.array(d) # Convert to numpy array
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    N = 1024
    fs = 44100 # sampling rate
    T = 1/float(fs)
    a=b=N/2
    nVec = np.arange(0,N*T,T)
    freqVec = np.arange(0,fs/2-(fs/N),fs/N)
    xn = np.cos((2*np.pi*2000*nVec))

    sinFFT = np.fft.fft(SineWindow(xn))
    hannFFT = np.fft.fft(HanningWindow(xn))
    sinMDCT = mdct.MDCT(SineWindow(xn),a,b)

    # Part 2f.
    sineWin = (1/float(N))*np.sum(np.power(SineWindow(np.ones_like(xn)),2)) # Get avg pow of sine window
    hannWin = (1/float(N))*np.sum(np.power(HanningWindow(np.ones_like(xn)),2)) # Get avg pow of hann window

    sinFFTdB = 96 + 10* np.log10((4/(np.power(N,2)*sineWin))*(sinFFT*np.conjugate(sinFFT)))
    hannFFTdB = 96 + 10* np.log10((4/(np.power(N,2)*hannWin))*(hannFFT*np.conjugate(hannFFT)))
    sinMDCTdB = 96 + 10* np.log10((2/(sineWin))*(sinMDCT*np.conjugate(sinMDCT)))
    ### YOUR TESTING CODE ENDS HERE ###

    plt.semilogx(freqVec,sinFFTdB[0:N/2],label='sinFFT',color='red')
    plt.hold(True)
    plt.plot(freqVec,hannFFTdB[0:N/2],label='hanFFT',color='green')
    plt.hold(True)
    plt.plot(freqVec+(fs/(2*N)),sinMDCTdB,label='sinMDCT')
    a = plt.gca()
    a.set_xlabel('Frequency (Hz)')
    a.set_ylabel('dB SPL')
    a.set_title('2f. Comparing Windows and Spectra')
    a.legend()

    plt.rcParams['figure.figsize'] = (8.0, 8.0)
    a.axis('tight')
    ### YOUR TESTING CODE ENDS HERE ###
