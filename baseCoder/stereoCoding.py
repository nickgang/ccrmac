"""
MS.py -- Module holding methods to implement M/S stereo coding

-----------------------------------------------------------------------
Copyright 2017 Ifueko Igbinedion -- All rights reserved
-----------------------------------------------------------------------

"""
import numpy as np
from psychoac import cbFreqLimits, AssignMDCTLinesFromFreqLimits 

couplingStart = 17 # Index of the frequency band we begin coupling

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
    couplingParams = []
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = float(mdctLineAssign[n])
        startIdx = sum(mdctLineAssign[:n])
        endIdx = int(startIdx + nLines)
        power_couple = sum(coupledChannel[startIdx:endIdx])/nLines
        for k in range(nChannels):
            power_chan = sum(np.array(coupleChans[k][startIdx:endIdx]))/nLines
            couplingParams.append(power_chan/power_couple)
    uncoupledData = np.array(uncoupledData)
    uncoupledPad = np.zeros((len(uncoupledData),nMdctLines-len(uncoupledData[0])))
    uncoupledData = np.hstack((uncoupledData,uncoupledPad))
    coupledPad = np.zeros((nMdctLines-len(coupledChannel)))
    coupledChannel = np.hstack((coupledPad,coupledChannel))
    return uncoupledData,coupledChannel,couplingParams

def ChannelDecoupling(uncoupledData,coupledChannel,couplingParams,Fs):
    nMDCTLines = len(coupledChannel) 
    nChannels = len(uncoupledData)
    couplingScales = couplingParams
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMDCTLines, Fs)
    uncoupledLen = sum(mdctLineAssign[:couplingStart])
    coupledLen = nMDCTLines-uncoupledLen 
    couplingIdx = 0
    startIdx = uncoupledLen
    reconstructedChannels = np.zeros([nChannels,nMDCTLines])
    if len(uncoupledData[0]) > 0:
        for k in range(nChannels):
            reconstructedChannels[k][:uncoupledLen] += uncoupledData[k][:uncoupledLen]
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = mdctLineAssign[n]
        endIdx = startIdx + nLines
        couplingBand = coupledChannel[startIdx:endIdx]
        for k in range(nChannels):
            scale = couplingScales[couplingIdx]
            couplingIdx += 1
            reconstructedChannels[k][startIdx:endIdx]+= scale*np.array(couplingBand)
        startIdx += nLines
    mdct = []
    for n in range(nChannels):
        mdct.append(list(reconstructedChannels[n]))
    return mdct

