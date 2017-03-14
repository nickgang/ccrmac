"""
SBR.py -- Module holding methods to implement Spectral Band Replication

-----------------------------------------------------------------------
Copyright 2017 Nick Gang -- All rights reserved
-----------------------------------------------------------------------
"""
import numpy as np
import psychoac as p
import window as w

########## Encoder Methods ##########

def calcSpecEnv(data,cutoff,fs,hfRecType=2):
    # calcSpecEnv - Function to calculate spectral envelope passed to decoder for SBR
    # data:      input signal for current block
    # cutoff:    Cutoff frequency in Hz
    # fs:        Sampling rate in Hz
    # hfRecType: 1 for full subband, 2 for altered subband that repeats top half
    N = data.size
    freqVec = np.arange(0,fs/2,fs/float(N)) # Vector of FFT bin frequencies
    Xn = np.fft.fft(w.HanningWindow(data),N)
    hannWin = (1/float(N))*np.sum(np.power(w.HanningWindow(np.ones_like(data)),2)) # Get avg pow of hann window
    XnI = (4/(np.power(N,2)*hannWin))*(np.power(np.abs(Xn),2)) # Compute values of FFT intensity
    # Transpose down top half of subband if we're using altered alg
    cutBin = freqToBinFFT(N,cutoff,fs)
    subBand = np.array(XnI[0:cutBin],copy=True)
    if hfRecType==2:
        try:
            subBand[0:int(np.floor(cutBin/2))] = subBand[int(np.floor(cutBin/2)):cutBin]
        except:
            subBand[0:int(np.floor(cutBin/2))] = subBand[int(np.floor(cutBin/2)+1):cutBin]
        # Add in additional high frequencies if there are less
        lowSize = int(len(subBand))
        highSize = int(len(XnI[cutBin:]))
        if lowSize<highSize:
            multiples = highSize/lowSize
            diff = highSize%lowSize
            addition = np.array(subBand,copy=True) # This is the piece to add in each time
            # Add in more copies of low band until sizes are comparable
            for m in range(multiples-1):
                subBand  = np.append(subBand,addition)
            if diff!=0:
                subBand = np.append(subBand,np.zeros(diff))
                subBand[-diff:] = subBand[0:diff]

    bandLimits = p.cbFreqLimits # Zwicker critical band upper limits
    cutBand = np.argwhere(bandLimits>=cutoff)[0] # Next band limit above cutoff freq
    nHfBands = len(bandLimits)-cutBand # How many bands will be reconstructed
    specEnv = np.zeros(nHfBands-1)

    for i in range(nHfBands-1):
        bandLines = np.intersect1d(np.argwhere(freqVec>bandLimits[cutBand+i]),\
                                   np.argwhere(freqVec<=bandLimits[cutBand+i+1]))
        highMean = np.mean(XnI[bandLines])
        subSlice = subBand[i*len(bandLines):(i+1)*len(bandLines)]
        subMean = np.mean(subSlice)
        # Spec Env is ratio of avg intensity in each hi-freq critical band to corresponding sub band
        specEnv[i]=highMean/subMean
    specEnv[np.nonzero(np.isnan(specEnv))] = 1 # Get rid of pesky nans
    return specEnv

########## Decoder Methods ##########

# High Frequency Reconstruction
def HiFreqRec(mdctLines,fs,cutoff,hfRecType=2):
    # hfRecType: 1 means straight bin transposition
    #            2 means altered transposition (default)
    nMDCT = len(mdctLines)
    cutBin = freqToBin(nMDCT,cutoff,fs)
    lowerBand = np.array(mdctLines[0:cutBin],copy=True)
    if hfRecType==1:
        mdctLines[cutBin+1:cutBin+len(lowerBand)+1] = lowerBand # Do the transposition
    elif hfRecType==2:
        try: # Being hacky to account for off by 1 errors
            lowerBand[0:int(np.floor(cutBin/2))] = lowerBand[int(np.floor(cutBin/2)):cutBin]
        except ValueError:
            lowerBand[0:int(np.floor(cutBin/2))] = lowerBand[int(np.floor(cutBin/2)+1):cutBin]
        # Add in additional high frequencies if reconstructed bin doesn't go all the way to nyquist
        lowSize = int(len(lowerBand))
        highSize = int(nMDCT-lowSize)
        if lowSize<highSize:
            multiples = highSize/lowSize
            diff = highSize%lowSize
            # Add in more copies of low band until sizes are comparable
            for m in range(multiples-1):
                np.append(lowerBand,lowerBand)
            if diff!=0:
                subBand = np.append(lowerBand,np.zeros(diff))
                lowerBand[-diff:] = lowerBand[0:diff]

        mdctLines[cutBin+1:cutBin+len(lowerBand)+1] = lowerBand # Do the transposition

    return mdctLines.astype(float) # If these are ints it can cause problems

# Additional High Frequency Components
def AddHiFreqs(mdctLines,fs,cutoff):
    nMDCT = len(mdctLines)
    cutBin = freqToBin(nMDCT,cutoff,fs)
    noiseBins = len(mdctLines[cutBin:])
    mdctLines[cutBin:] *= np.absolute(np.random.normal(1,0.5,noiseBins))# Add some noise to reconstructed bins
    return mdctLines

# Envelope Adjustment (assumes HF Reconstruction has occured)
def EnvAdjust(mdctLines,fs,cutoff,envelope):
    nMDCT = len(mdctLines)
    N = 2*nMDCT
    mdctFreq = np.arange(0,fs/2,fs/float(N))+(fs/float(2.*N))
    bandLimits = p.cbFreqLimits # Zwicker critical band upper limits
    cutBand = np.argwhere(bandLimits>=cutoff)[0] # Next band limit above cutoff freq
    nHfBands = len(envelope)
    tempLines = np.copy(mdctLines)

    for i in range(nHfBands):
        # Find MDCT lines in this critical band and apply envelope from encoder
        bandLines = np.intersect1d(np.argwhere(mdctFreq>bandLimits[cutBand+i]),\
                                   np.argwhere(mdctFreq<=bandLimits[cutBand+i+1]))
        bandInt = np.mean(tempLines[bandLines]) # Average intensity in current band
        tempLines[bandLines] *= envelope[i]
    return tempLines

# Utility Function to Convert Cutoff Freq to Bin number
def freqToBin(nMDCT,cutoff,fs):
    N = 2*nMDCT
    freqVec = np.arange(0,fs/2,fs/float(N))+fs/float(2.*N) # MDCT Frequencies
    cutBin = np.argmin(np.absolute(freqVec-cutoff)) # Find index of cutoff frequency
    return cutBin

# Utility Function to Convert Cutoff Freq to FFT Bin Number
def freqToBinFFT(nFFT,cutoff,fs):
    N = nFFT
    freqVec = np.arange(0,fs/2,fs/float(N)) # FFT freq vector
    cutBin = np.argmin(np.absolute(freqVec-cutoff))
    return cutBin

# Utility Function to Convert cutoff frequency to critical band index
def freqToBand(cutoff):
    bandLimits = p.cbFreqLimits
    cutBand = np.argwhere(bandLimits>=cutoff)[0] # Next band limit above cutoff freq
    return cutBand
