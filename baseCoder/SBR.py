"""
SBR.py -- Module holding methods to implement Spectral Band Replication

-----------------------------------------------------------------------
Copyright 2017 Nick Gang -- All rights reserved
-----------------------------------------------------------------------
"""
import numpy as np

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
    mdctLines[cutBin:] *= np.absolute(np.random.normal(1,0.1,noiseBins)) # Add some noise to reconstructed bins
    return mdctLines

# Envelope Adjustment
def EnvAdjust(mdctLines,fs,envelope):
    return np.zeros_like(mdctLines)

# Utility Function to Convert Cutoff Freq to Bin number
def freqToBin(nMDCT,cutoff,fs):
    N = 2*nMDCT
    freqVec = np.arange(0,fs/2,fs/float(N))+fs/float(2.*N) # MDCT Frequencies
    cutBin = np.argmin(np.absolute(freqVec-cutoff)) # Find index of cutoff frequency
    return cutBin
