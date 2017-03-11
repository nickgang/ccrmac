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
from stereoCoding import * # Methods for stereo coding

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc,BitAllocUniform,BitAllocConstSNR,BitAllocConstMNR  #allocates bits to scale factor bands given SMRs
from scipy import signal # signal processing tools


def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,couplingParams,codingParams):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level

    if codingParams.doSBR == True:
        ### SBR Decoder Module 1 - High Frequency Reconstruction ###
        mdctLine = HiFreqRec(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)

        ### SBR Decoder Module 2 - Additional High Frequency Components ###
        mdctLine = AddHiFreqs(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)
        ### SBR Decoder Module 3 - Envelope Adjustment ###

    # IMDCT and window the data for this channel
    # data = SineWindow( IMDCT(mdctLine, halfN, halfN) )  # takes in halfN MDCT coeffs
    mdcTLine = ChannelDecoupling([[],[]],mdctLine,couplingParams,codingParams.sampleRate)
    data = []
    for k in range(codingParams.nChannels):
        data.append(SineWindow( IMDCT(mdctLine[k], halfN, halfN) ))  # takes in halfN MDCT coeffs

    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    # (STEREO CODING)
    #data = MSEncode(data)
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    couplingParams = []
    # loop over channels and separately encode each one
    for iCh in range(1):
        (s,b,m,o,c) = EncodeDataWithCoupling(data,codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
        couplingParams = c
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor,couplingParams)

def EncodeDataWithCoupling(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
 
    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN * codingParams.nChannels  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    # Calculate Spectral Envelope based on original signal
    # specEnv = calcSpecEnv(data,codingParams.sbrCutoff,codingParams.sampleRate)

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    #Decimate and lowpass signal by factor determined by cutoff frequency
    doDecimate = False

    #TODO: grab previous and next block for filtering
    if doDecimate==True:
        Wc = codingParams.sbrCutoff/float(codingParams.sampleRate/2.)# Normalized cutoff frequency
        b,a = butter(4,Wn)
        data = lfilter(b,a,data)
    mdctLines = []
    for k in range(codingParams.nChannels):
        mdctTimeSamples = SineWindow(data[k])
        mdctLines.append(MDCT(mdctTimeSamples,halfN,halfN)[:halfN])
    print mdctLines
    couplingParams = []
    emptyArray,mdctLines,couplingParams = ChannelCoupling(mdctLines, codingParams.sampleRate)
    print mdctLines
    mdctLines = np.array(mdctLines)
    bitBudget -= codingParams.nCouplingScaleBits*len(couplingParams) # less bits for coupling scale params

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    print mdctLines.size()
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale, couplingParams)

def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN = codingParams.nMDCTLines
    N = 2*halfN
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands = codingParams.sfBands
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)

    # compute target mantissa bit budget for this block of halfN MDCT mantissas
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands +1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits

    # Calculate Spectral Envelope based on original signal
    # specEnv = calcSpecEnv(data,codingParams.sbrCutoff,codingParams.sampleRate)

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    #Decimate and lowpass signal by factor determined by cutoff frequency
    doDecimate = False

    #TODO: grab previous and next block for filtering
    if doDecimate==True:
        Wc = codingParams.sbrCutoff/float(codingParams.sampleRate/2.)# Normalized cutoff frequency
        b,a = butter(4,Wn)
        data = lfilter(b,a,data)

    mdctTimeSamples = SineWindow(data)
    mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)
    # perform bit allocation using SMR results
    bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant=halfN
    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)
