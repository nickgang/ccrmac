"""
SBR.py -- Module holding methods to implement Spectral Band Replication

-----------------------------------------------------------------------
© 2017 Nick Gang -- All rights reserved
-----------------------------------------------------------------------

"""
import numpy as np

# High Frequency Reconstruction
def HiFreqRec(mdctLines,fs,cutoff):
    nMDCT = len(mdctLines)
    N = 2*nMDCT
    freqVec = np.arange(0,fs/2,fs/float(N))+fs/float(2.*N) # MDCT Frequencies
    cutBin = np.argmin(np.absolute(freqVec-cutoff)) # Find index of cutoff frequency
    lowerBand = mdctLines[0:cutBin]
    mdctLines[cutBin+1:cutBin+len(lowerBand)+1] = lowerBand # Do the transposition
    return mdctLines

# Additional High Frequency Components
def AddHiFreqs(mdctLines,fs):
    return np.zeros_like(mdctLines)

# Envelope Adjustment
def EnvAdjust(mdctLines,fs,envelope):
    return np.zeros_like(mdctLines)
