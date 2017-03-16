"""
MS.py -- Module holding methods to implement M/S stereo coding

-----------------------------------------------------------------------
Copyright 2017 Ifueko Igbinedion -- All rights reserved
-----------------------------------------------------------------------

"""
import numpy as np
from psychoac import cbFreqLimits, AssignMDCTLinesFromFreqLimits 

def MSEncode(data, couplingStart, sfBands):
    end = int(sum(sfBands.nLines[:couplingStart]))
    data = np.array(data)
    L = data[0][:end]
    R = data[1][:end]
    data[0][:end] = (L + R)/2
    data[1][:end] = (L - R)/2
    return data

def MSDecode(data, couplingStart, sfBands):
    end = int(sum(sfBands.nLines[:couplingStart]))
    data = np.array(data)
    Lp = data[0][:end]
    Rp = data[1][:end]
    data[0][:end] = Lp + Rp
    data[1][:end] = Lp - Rp
    return data

def ChannelCoupling(mdct, Fs, couplingStart):
    """
    Takes a multi-channel mdct and couples channels at
    high frequencies
    """
    # TODO: use spectral power to compute start of coupling
    #mdct[0] = np.roll(mdct[0],len(mdct[0])/2)
    nChannels = len(mdct)
    nMdctLines = len(mdct[0]) 
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMdctLines, Fs)
    start = int(sum(mdctLineAssign[:couplingStart])-1)
    coupledChannel = np.zeros(nMdctLines - start)
    uncoupledData = []
    coupleChans = []
    for n in range(nChannels):
        coupleChans.append(mdct[n][start:])
        coupledChannel += np.array(mdct[n][start:])
        uncoupledData.append(mdct[n][:start])
    #coupledChannel /= nChannels
    couplingParams = np.zeros(1+(len(mdctLineAssign)-couplingStart)*nChannels).astype(float)
    couplingParams[0] = 0
    couplingIdx = 1
    
    for n in range(couplingStart,len(mdctLineAssign)):
        
        nLines = int(mdctLineAssign[n])
        startIdx = int(sum(mdctLineAssign[:n])-sum(mdctLineAssign[:couplingStart]))
        endIdx = int(startIdx + nLines) 
        power_couple = np.sum(coupledChannel[startIdx:endIdx]**2)/float(nLines)
        for k in range(nChannels):
            power_chan = np.sum(np.array(coupleChans[k][startIdx:endIdx])**2)/float(nLines)
            if power_couple:
                couplingParams[couplingIdx] = np.sqrt(power_chan/power_couple)
            couplingIdx +=1
            
            
    uncoupledData = np.array(uncoupledData)
    uncoupledPad = np.zeros((len(uncoupledData),nMdctLines-len(uncoupledData[0])))
    uncoupledData = np.hstack((uncoupledData,uncoupledPad))
    coupledPad = np.zeros((nMdctLines-len(coupledChannel)))
    coupledChannel = np.hstack((coupledPad,coupledChannel))
    #print coupledChannel[-10:]
    #print "coupling params are", couplingParams
    return uncoupledData,coupledChannel,couplingParams

def ChannelDecoupling(uncoupledData,coupledChannel,couplingParams,Fs,couplingStart):
    nMDCTLines = len(coupledChannel) 
    nChannels = len(uncoupledData)
    couplingScales = couplingParams
    #print couplingParams
    #print coupledChannel[-10:]
    # this is in codingParams, would it be better to just pass codingParams to the function
    # and call codingParams.sfBands.nLines?  would also have access to codingParams.nMDCTLines
    mdctLineAssign = AssignMDCTLinesFromFreqLimits(nMDCTLines, Fs) 
    uncoupledLen = int(sum(mdctLineAssign[:couplingStart]))
    coupledLen = int(nMDCTLines-uncoupledLen)
    couplingIdx = 1
    startIdx = int(uncoupledLen)
    reconstructedChannels = np.zeros([nChannels,nMDCTLines])
    if len(uncoupledData[0]) > 0:
        for k in range(nChannels):
            reconstructedChannels[k][:uncoupledLen] += uncoupledData[k][:uncoupledLen]
    for n in range(couplingStart,len(mdctLineAssign)):
        nLines = mdctLineAssign[n]
        endIdx = int(startIdx + nLines)
        #print startIdx,endIdx
        couplingBand = coupledChannel[startIdx:endIdx]
        for k in range(nChannels):
            scale = couplingScales[couplingIdx]
            couplingIdx += 1
            #print scale
            reconstructedChannels[k][startIdx:endIdx]+= scale*np.array(couplingBand)
        startIdx += int(nLines)
    mdct = []
    for n in range(nChannels):
        mdct.append(list(reconstructedChannels[n]))
    return mdct

