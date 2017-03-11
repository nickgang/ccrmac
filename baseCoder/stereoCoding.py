"""
MS.py -- Module holding methods to implement M/S stereo coding

-----------------------------------------------------------------------
Copyright 2017 Ifueko Igbinedion -- All rights reserved
-----------------------------------------------------------------------

"""
import numpy as np
from psychoac import cbFreqLimits, AssignMDCTLinesFromFreqLimits 

def MSEncode(data):
    L = data[0]
    R = data[1]
    data[0] = (L + R)/2
    data[1] = (L - R)/2
    return data

def MSDecode(data):
    Lp = data[0]
    Rp = data[1]
    data[0] = Lp + Rp
    data[1] = Lp - Rp
    return data

def ChannelCoupling(mdct, Fs):
    """
    Takes a multi-channel mdct and couples channels at
    high frequencies
    """
    # TODO: use spectral power to compute start of coupling
    couplingStart = 0 # Index of the frequency band we begin coupling
    nChannels = len(mdct)
    nMdctLines = len(mdct[0]) 
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMdctLines, Fs)
    start = sum(mdctLineAssign[:couplingStart])
    coupledChannel = np.zeros(nMdctLines - start)
    uncoupledData = []
    coupleChans = []
    for n in range(nChannels):
        coupleChans.append(mdct[n][start:])
        # TODO: Phase adjustment
        coupledChannel += mdct[n][start:]
        uncoupledData.append(mdct[n][:start])
    couplingParams = [couplingStart]
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = float(mdctLineAssign[n])
        startIdx = sum(mdctLineAssign[:n])
        endIdx = int(startIdx + nLines)
        power_couple = sum(coupledChannel[startIdx:endIdx])/nLines
        for k in range(nChannels):
            power_chan = sum(np.array(coupleChans[k][startIdx:endIdx]))/nLines
            couplingParams.append(power_chan/power_couple)
    return uncoupledData,list(coupledChannel),couplingParams

def ChannelDecoupling(uncoupledData,coupledChannel,couplingParams,Fs):
    uncoupledLen = len(uncoupledData[0])
    coupledLen = len(coupledChannel)
    nMDCTLines = uncoupledLen + coupledLen
    nChannels = len(uncoupledData)
    couplingStart = couplingParams[0]
    couplingScales = couplingParams[1:]
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMDCTLines, Fs)
    couplingIdx = 0
    startIdx = 0
    reconstructedChannels = np.zeros([nChannels,nMDCTLines])
    if len(uncoupledData[0]) > 0:
        for k in range(nChannels):
            reconstructedChannels[k][:uncoupledLen] += uncoupledData[k]
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = mdctLineAssign[n]
        endIdx = startIdx + nLines
        couplingBand = coupledChannel[startIdx:endIdx]
        for k in range(nChannels):
            scale = couplingScales[couplingIdx]
            couplingIdx += 1
            reconstructedChannels[k][uncoupledLen+startIdx:uncoupledLen+endIdx]+= scale*np.array(couplingBand)
        startIdx += nLines
    mdct = []
    for n in range(nChannels):
        mdct.append(list(reconstructedChannels[n]))
    return mdct

