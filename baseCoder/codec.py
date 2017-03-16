"""
codec.py -- The actual encode/decode functions for the perceptual audio codec
-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow,KBDWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)
from SBR import * # Methods used for SBR
from stereoCoding import *

# used only by Encode
from psychoac import *  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc,BitAllocSBR  #allocates bits to scale factor bands given SMRs
from scipy import signal # signal processing tools

#SHORTBLOCKSIZE = 256
#LONGBLOCKSIZE = 2048

def Decode(scaleFactorFull,bitAllocFull,mantissaFull,overallScaleFactorFull,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    if(codingParams.blocksize == 3):
        #print "MDCTLines: ", codingParams.nMDCTLines
        a = codingParams.longBlockSize/2
        b = codingParams.shortBlockSize/2
    elif (codingParams.blocksize == 2):
        a = codingParams.shortBlockSize/2
        b = a
    elif (codingParams.blocksize == 1):
        b = codingParams.longBlockSize/2
        a = codingParams.shortBlockSize/2
    else:
        a = codingParams.longBlockSize/2
        b = a
    N = a+b
    halfN = N/2

    #halfN = codingParams.nMDCTLines
    #N = 2*halfN
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)
    data = []
    mdctLines = []
    for iCh in range(codingParams.nChannels):

        scaleFactor = scaleFactorFull[iCh]
        bitAlloc = bitAllocFull[iCh]
        mantissa = mantissaFull[iCh]
        overallScaleFactor = overallScaleFactorFull[iCh]
        rescaleLevel = 1.*(1<<overallScaleFactorFull[iCh])
        # reconstitute the first halfN MDCT lines of this channel from the stored data
        mdctLine = np.zeros(halfN,dtype=np.float64)
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines =codingParams.sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel  # put overall gain back to original level
        mdctLines.append(mdctLine)

    #print codingParams.couplingParams
    if codingParams.doCoupling == True and len(mdctLines[0]) > 128:
        #print len(mdctLines[0])
        mdctLines = np.array(mdctLines)
        # better to just pass codingParams to channelDecoupling?
        mdctLines = ChannelDecoupling(mdctLines,codingParams.coupledChannel,codingParams.couplingParams,codingParams.sampleRate,codingParams.nCouplingStart)

    mdctLines = np.array(mdctLines)
    for iCh in range(codingParams.nChannels):
        data.append(np.array([],dtype=np.float64))  # add location for this channel's data
        mdctLine = mdctLines[iCh]
        if codingParams.doSBR == True:
            ### SBR Decoder Module 1 - High Frequency Reconstruction ###
            mdctLine = HiFreqRec(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)
            ### SBR Decoder Module 2 - Additional High Frequency Components ###
            mdctLine = AddHiFreqs(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)
            ### SBR Decoder Module 3 - Envelope Adjustment ###
            mdctLine = EnvAdjust(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff,codingParams.specEnv[iCh])
            # print codingParams.specEnv # Print envelope for debugging purposes

        # IMDCT and window the data for this channel
        # data = SineWindow( IMDCT(mdctLine, halfN, halfN) )  # takes in halfN MDCT coeffs
        imdct = IMDCT(mdctLine, a, b)   # takes in halfN MDCT coeffs
        data[iCh] = np.append(SineWindow(np.append(imdct[:a],np.zeros(a)))[:a],SineWindow(np.append(np.zeros(b),imdct[a:]))[b:])
        #print data.size
    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)

    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    codingParams.couplingParams = np.zeros(1+((25-codingParams.nCouplingStart)*codingParams.nChannels)).astype(float)
    codingParams.coupledChannel = np.ones(codingParams.nMDCTLines,np.float64)*0.5
    if codingParams.doCoupling == True and codingParams.nMDCTLines > 128:
        (scaleFactor,bitAlloc,mantissa,overallScaleFactor) = EncodeDataWithCoupling(np.array(data),codingParams)
    else:
        # loop over channels and separately encode each one
        for iCh in range(codingParams.nChannels):
            (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams,iCh)
            scaleFactor.append(s)
            bitAlloc.append(b)
            mantissa.append(m)
            overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor)

def EncodeDataWithCoupling(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""
    # NEW: Determine block type and set a,b
    if(codingParams.blocksize < 2):
        b = codingParams.longBlockSize/2
    else:
        b = codingParams.shortBlockSize/2
    if(codingParams.blocksize == 1 or codingParams.blocksize == 2):
        a = codingParams.shortBlockSize/2
    else:
        a = codingParams.longBlockSize/2
    N = a+b
    halfN = N/2
    #print "A: ", a
    #print "B: ", b
    #print "halfN: ", halfN

    # Reclaim nScaleBits from bands with 0 lines
    # vary bark width of bands
    # pass different flimits to AssignMDCTLines...

    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)
    sfBands = codingParams.sfBands
    # db print "Encode coupling: ", sfBands.nLines
    # NEW compute target bit rate based on block type
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands + 1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    bitBudget -= 2 # block ID size TODO: make this a variable
    mdctLinesFull = []
    for iCh in range(codingParams.nChannels):
        if codingParams.doSBR == True:
            # Calculate Spectral Envelope based on original signal
            specEnv = calcSpecEnv(data[iCh],codingParams.sbrCutoff,codingParams.sampleRate)
            # Append in spectral envelope for this channel into empty container
            codingParams.specEnv[iCh][:] = specEnv

            #Decimate and lowpass signal by factor determined by cutoff frequency
            doDecimate = False
            if doDecimate==True:
                Wc = codingParams.sbrCutoff/float(codingParams.sampleRate/2.)# Normalized cutoff frequency
                B,A = signal.butter(4,Wn)
                data[iCh] = signal.lfilter(B,A,data[iCh])

        # window data for side chain FFT and also window and compute MDCT
        timeSamples = data[iCh]
        # Window data based on block size
        mdctTimeSamples = np.append(SineWindow(np.append(timeSamples[:a],np.zeros(a)))[:a],SineWindow(np.append(np.zeros(b),timeSamples[a:]))[b:])
        # Call MDCT with a, b reflecting block size
        mdctLines = MDCT(mdctTimeSamples, a, b)

        # compute overall scale factor for this block and boost mdctLines using it
        maxLine = np.max( np.abs(mdctLines) )
        overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroaes don't depend on nMantBits
        mdctLines *= (1<<overallScale)
        mdctLinesFull.append(mdctLines)

    uncoupledData, coupledChannel, couplingParams = ChannelCoupling(mdctLinesFull,codingParams.sampleRate,codingParams.nCouplingStart)
    codingParams.couplingParams = couplingParams
    codingParams.coupledChannel = coupledChannel
    mdctLinesFull = uncoupledData

    scaleFactorFull= []
    bitAllocFull = []
    mantissaFull = []
    overallScaleFull = []
    for iCh in range(codingParams.nChannels):
        # compute the mantissa bit allocations
        # compute SMRs in side chain FFT
        SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
        if codingParams.doSBR == True:
            # Critical band starting here are above cutoff
            cutBin = freqToBand(codingParams.sbrCutoff)
            # perform bit allocation using SMR results
            bitAlloc = BitAllocSBR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir, codingParams.blocksize, cutBin)
        else:
            bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir, codingParams.blocksize)
        codingParams.bitReservoir += bitBudget - np.sum(bitAlloc * sfBands.nLines)
        # db print "blocksize: ", codingParams.blocksize
        # db print "Bit Reservoir: ", codingParams.bitReservoir
        # db if codingParams.blocksize == 2:
        # db     print bitAlloc
        # given the bit allocations, quantize the mdct lines in each band
        scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
        nMant = halfN

        for iBand in range(sfBands.nBands):
            if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
        mantissa=np.empty(nMant,dtype=np.int32)
        iMant=0
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
            nLines= sfBands.nLines[iBand]
            if(highLine - lowLine > 0):
                scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
            else:
                scaleLine = abs(mdctLines[lowLine])
            scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
            if bitAlloc[iBand]:
                mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                iMant += nLines
        # end of loop over scale factor bands
        scaleFactorFull.append(scaleFactor)
        bitAllocFull.append(bitAlloc)
        mantissaFull.append(mantissa)
        overallScaleFull.append(overallScale)
        # return results
    return (scaleFactorFull, bitAllocFull, mantissaFull, overallScaleFull)


def EncodeSingleChannel(data,codingParams,iCh):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""
    # NEW: Determine block type and set a,b
    if(codingParams.blocksize < 2):
        b = codingParams.longBlockSize/2
    else:
        b = codingParams.shortBlockSize/2
    if(codingParams.blocksize == 1 or codingParams.blocksize == 2):
        a = codingParams.shortBlockSize/2
    else:
        a = codingParams.longBlockSize/2
    N = a+b
    halfN = N/2
    #print "A: ", a
    #print "B: ", b
    #print "halfN: ", halfN

    # Reclaim nScaleBits from bands with 0 lines
    # vary bark width of bands
    # pass different flimits to AssignMDCTLines...

    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)
    sfBands = codingParams.sfBands

    # NEW compute target bit rate based on block type
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands + 1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    bitBudget -= 2 # block ID size TODO: make this a variable

    if codingParams.doSBR == True:
        # Calculate Spectral Envelope based on original signal
        specEnv = calcSpecEnv(data,codingParams.sbrCutoff,codingParams.sampleRate)
        # Append in spectral envelope for this channel into empty container
        codingParams.specEnv[iCh][:] = specEnv

        #Decimate and lowpass signal by factor determined by cutoff frequency
        doDecimate = False
        if doDecimate==True:
            Wc = codingParams.sbrCutoff/float(codingParams.sampleRate/2.)# Normalized cutoff frequency
            B,A = signal.butter(4,Wn)
            data = signal.lfilter(B,A,data)

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    # Window data based on block size
    mdctTimeSamples = np.append(SineWindow(np.append(timeSamples[:a],np.zeros(a)))[:a],SineWindow(np.append(np.zeros(b),timeSamples[a:]))[b:])
    # Call MDCT with a, b reflecting block size
    mdctLines = MDCT(mdctTimeSamples, a, b)

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    if codingParams.doSBR == True:
        # Critical band starting here are above cutoff
        cutBin = freqToBand(codingParams.sbrCutoff)
        # perform bit allocation using SMR results
        bitAlloc = BitAllocSBR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir, codingParams.blocksize, cutBin)
    else:
        bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs, codingParams.bitReservoir, codingParams.blocksize)
    codingParams.bitReservoir += bitBudget - np.sum(bitAlloc * sfBands.nLines)
    #print "blocksize: ", codingParams.blocksize
    #print "Bit Reservoir: ", codingParams.bitReservoir
    #if codingParams.blocksize == 2:
    #   print bitAlloc
    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant = halfN

    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        if(highLine - lowLine > 0):
            scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        else:
            scaleLine = abs(mdctLines[lowLine])
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)