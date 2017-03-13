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

def ChannelCoupling(mdct, Fs, couplingStart):
    """
    Takes a multi-channel mdct and couples channels at
    high frequencies
    """
    # TODO: use spectral power to compute start of coupling
    mdct = MSEncode(np.array(mdct))
    mdct[0] = np.roll(mdct[0],len(mdct[0])/2)
    nChannels = len(mdct)
    nMdctLines = len(mdct[0]) 
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMdctLines, Fs)
    start = int(sum(mdctLineAssign[:couplingStart]))
    coupledChannel = np.zeros(nMdctLines - start)
    uncoupledData = []
    coupleChans = []
    phase_shift = 0
    for n in range(nChannels):
        phase_shift = np.min((np.min(mdct[n]),phase_shift))
    phase_shift *= -1.0
    phase_shift = 0.0
    for n in range(nChannels):
        coupleChans.append(mdct[n][start:])
        # TODO: Phase adjustment
        coupledChannel += np.array(mdct[n][start:])+phase_shift
        uncoupledData.append(mdct[n][:start])
    couplingParams = np.zeros(1+(len(mdctLineAssign)-couplingStart)*nChannels).astype(float)
    couplingParams[0] = phase_shift
    couplingIdx = 1
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = float(mdctLineAssign[n])
        startIdx = sum(mdctLineAssign[:n])
        endIdx = int(startIdx + nLines)
        power_couple = sum(coupledChannel[startIdx:endIdx])/nLines
        for k in range(nChannels):
            power_chan = sum(np.array(coupleChans[k][startIdx:endIdx])+phase_shift)/float(nLines)
            if power_couple > 0:
                couplingParams[couplingIdx] = (power_chan/power_couple)
            couplingIdx +=1
            
    uncoupledData = np.array(uncoupledData)
    uncoupledPad = np.zeros((len(uncoupledData),nMdctLines-len(uncoupledData[0])))
    uncoupledData = np.hstack((uncoupledData,uncoupledPad))
    coupledPad = np.zeros((nMdctLines-len(coupledChannel)))
    coupledChannel = np.hstack((coupledPad,coupledChannel))
    return uncoupledData,coupledChannel,couplingParams

def ChannelDecoupling(uncoupledData,coupledChannel,couplingParams,Fs,couplingStart):
    nMDCTLines = len(coupledChannel) 
    nChannels = len(uncoupledData)
    couplingScales = couplingParams
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMDCTLines, Fs)
    uncoupledLen = sum(mdctLineAssign[:couplingStart])
    coupledLen = nMDCTLines-uncoupledLen 
    phase_shift = couplingParams[0]
    couplingIdx = 1
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
            reconstructedChannels[k][startIdx:endIdx]+= scale*np.array(couplingBand)-phase_shift
        startIdx += nLines
    mdct = []
    for n in range(nChannels):
        mdct.append(list(reconstructedChannels[n]))
    mdct[0] = np.roll(mdct[0],-len(mdct[0])/2)
    mdct = MSDecode(np.array(mdct))
    return mdct

