"""
SBR.py -- Module holding methods to implement Spectral Band Replication

-----------------------------------------------------------------------
Copyright 2017 Nick Gang -- All rights reserved
-----------------------------------------------------------------------
"""
import numpy as np
import window as w

########## Encoder Methods ##########

def calcSpecEnv(data,cutoff,fs):
    # calcSpecEnv - Function to calculate spectral envelope passed to decoder for SBR
    # data:      input signal for current block
    # cutoff:    Cutoff frequency in Hz
    # fs:        Sampling rate in Hz

    N = data.size
    freqVec = np.arange(0,fs/2,fs/float(N)) # Vector of FFT bin frequencies
    Xn = np.fft.fft(w.HanningWindow(data),N)
    hannWin = (1/float(N))*np.sum(np.power(w.HanningWindow(np.ones_like(data)),2)) # Get avg pow of hann window
    XnI = (4/(np.power(N,2)*hannWin))*(np.power(np.abs(Xn),2)) # Compute values of FFT intensity
    bandLimits = p.cbFreqLimits # Zwicker critical band upper limits
    cutBand = np.argwhere(bandLimits>=cutoff)[0] # Next band limit above cutoff freq
    nHfBands = len(bandLimits)-cutBand # How many bands will be reconstructed
    specEnv = np.zeros(nHfBands-1)
    for i in range(nHfBands-1):
        bandLines = np.intersect1d(np.argwhere(freqVec>bandLimits[cutBand+i]),np.argwhere(freqVec<=bandLimits[cutBand+i+1]))
        specEnv[i] = np.mean(XnI[bandLines]) # Spec Env is avg intensity in each hi-freq critical band

    return specEnv

########## Decoder Methods ##########

# High Frequency Reconstruction
def HiFreqRec(mdctLines,fs,cutoff):
    nMDCT = len(mdctLines)
    cutBin = freqToBin(nMDCT,cutoff,fs)
    lowerBand = mdctLines[0:cutBin]
    mdctLines[cutBin+1:cutBin+len(lowerBand)+1] = lowerBand # Do the transposition
    return mdctLines.astype(float) # If these are ints it can cause problems

# Additional High Frequency Components
def AddHiFreqs(mdctLines,fs,cutoff):
    nMDCT = len(mdctLines)
    cutBin = freqToBin(nMDCT,cutoff,fs)
    noiseBins = len(mdctLines[cutBin:])
    mdctLines[cutBin:] *= np.absolute(np.random.normal(1,0.5,noiseBins))# Add some noise to reconstructed bins
    mdctLines[cutBin:] *= 0.1 # Hardcoding this hack until envelope is ready
    return mdctLines

# Envelope Adjustment
# Envelope Adjustment
def EnvAdjust(mdctLines,fs,envelope):
    nMDCT = len(mdctLines)
    mdctFreq = np.arange(0,fs/2,fs/float(N))+(fs/float(2.*N))
    bandLimits = p.cbFreqLimits # Zwicker critical band upper limits
    cutBand = np.argwhere(bandLimits>=cutoff)[0] # Next band limit above cutoff freq
    nHfBands = len(envelope)
    tempLines = np.copy(mdctLines)

    for i in range(nHfBands):
        # Find MDCT lines in this critical band and apply envelope from encoder
        bandLines = np.intersect1d(np.argwhere(mdctFreq>bandLimits[cutBand+i]),\
                                   np.argwhere(mdctFreq<=bandLimits[cutBand+i+1]))
        tempLines[bandLines] *= envelope[i]

    return tempLines

# Utility Function to Convert Cutoff Freq to Bin number
def freqToBin(nMDCT,cutoff,fs):
    N = 2*nMDCT
    freqVec = np.arange(0,fs/2,fs/float(N))+fs/float(2.*N) # MDCT Frequencies
    cutBin = np.argmin(np.absolute(freqVec-cutoff)) # Find index of cutoff frequency
    return cutBin
