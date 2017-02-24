"""
quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")

Author: Nick Gang
Date: 1/27/17
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    xMax = 1 # Overload level
    sign = 0 # Initialize sign
    absVal = np.absolute(aNum) # Store absolute value

    # Set sign bit (don't use negative zero)
    if aNum >= 0:
        sign = np.left_shift(0,nBits-1)
    else :
        sign = np.left_shift(1,nBits-1)

    if absVal >= 1:
        code = np.power(2,nBits-1)-1
    else :
        preCode = ((np.power(2,nBits)-1)*absVal + 1)/2 # Code number to be truncated
        code = np.trunc(preCode)
        code = code.astype(int) # Cast truncated number to integer

    aQuantizedNum = sign + code
    ### YOUR CODE ENDS HERE ###
    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    ### YOUR CODE STARTS HERE ###
    signMask = np.left_shift(1,nBits-1)
    neg = np.right_shift(np.bitwise_and(signMask,aQuantizedNum),nBits-1)
    sign = np.power(-1,neg) # Extract sign from coded value

    code = np.bitwise_xor(np.bitwise_or(aQuantizedNum,signMask),signMask) # strip sign bit to get just |code|
    number = (2*code.astype(float))/(np.power(2,nBits)-1)
    aNum = sign*number
    ### YOUR CODE ENDS HERE ###
    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """
    #Notes:
    #Make sure to vectorize your function properly, as specified in the homework instructions

    ### YOUR CODE STARTS HERE ###
    xMax = 1 # Overload level
    sign = np.zeros_like(aNumVec, dtype=int) # Initialize sign vector
    absVal = np.absolute(aNumVec) # Store absolute values of inputs

    # Set sign bit (don't use negative zero)
    sign = np.left_shift(1-(aNumVec>=0),nBits-1)

    code = np.zeros_like(aNumVec, dtype=int)
    code[np.nonzero(absVal >= 1)] = np.power(2,nBits-1)-1 # Set overflow values to all ones

    preCode = ((np.power(2,nBits)-1)*absVal + 1)/2 # Code number to be truncated
    code[np.nonzero(absVal<1)] = np.trunc(preCode[np.nonzero(absVal<1)])

    code = code.astype(int) # Cast truncated number to integer (numpy.trunc returns float)

    aQuantizedNumVec = sign + code
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """
    ### YOUR CODE STARTS HERE ###
    signMask = np.left_shift(np.ones_like(aQuantizedNumVec),nBits-1)
    neg = np.right_shift(np.bitwise_and(signMask,aQuantizedNumVec),nBits-1) # are codes negative or not
    sign = np.power(-1,neg) # Extract sign values from sign bits

    code = np.bitwise_xor(np.bitwise_or(aQuantizedNumVec,signMask),signMask) # strip sign bit to get just |code|
    number = (2*code.astype(float))/(np.power(2,nBits)-1)
    aNumVec = sign*number
    ### YOUR CODE ENDS HERE ###

    return aNumVec

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros

    ### YOUR CODE STARTS HERE ###
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits
    codeEq = QuantizeUniform(aNum,nBitsEq)

    signMask = np.left_shift(1,nBitsEq-1)
    neg = np.right_shift(np.bitwise_and(signMask,codeEq),nBitsEq-1)
    sign = np.power(-1,neg) # Extract sign from coded value

    code = np.bitwise_xor(np.bitwise_or(codeEq,signMask),signMask) # strip sign bit to get just |code|

    scaleMax = np.power(2,nScaleBits)-1 # Maximum number of leading zeros we can store
    count = nBitsEq-1

    while code > 0 and count > 0:
        code = np.right_shift(code,1)
        count -= 1

    if count >= scaleMax:
        scale = scaleMax
    else:
        scale = count
    ### YOUR CODE ENDS HERE ###

    return scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits
    codeEq = QuantizeUniform(aNum,nBitsEq)

    signMask = np.left_shift(1,nBitsEq-1)
    signBit = np.right_shift(np.bitwise_and(signMask,codeEq),nBitsEq-1) # This gives value of sign bit
    sign = np.left_shift(signBit,nMantBits-1) # Stick sign bit in first position of mantissa

    code = np.bitwise_xor(np.bitwise_or(codeEq,signMask),signMask) # strip sign bit to get just |code|

    scaleMax = np.power(2,nScaleBits)-1 # Maximum value of scale bit

    if scale == scaleMax:
        mantissa = sign + code # Max leading 0's, Leave in leading 1
    else:
        onesMask = np.power(2,nMantBits)-1 # All ones
        leadingBit = np.left_shift(1,nMantBits-1) # Just leading bit (msb of )
        leadingOneMask = np.bitwise_xor(onesMask,leadingBit)

        shiftCode = np.right_shift(code,(nBitsEq-1)-nMantBits-scale) # Bit shift larger code so it fits in mantissa
        noLeadingOne = np.bitwise_and(leadingOneMask,shiftCode)
        mantissa = sign + np.bitwise_and(leadingOneMask,shiftCode)  # Omit leading 1

    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    scaleMax = np.power(2,nScaleBits)-1
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits

    # Get sign bit from mantissa and put it in highest order bit
    signBit = np.right_shift(mantissa,nMantBits-1)
    sign = np.left_shift(signBit,nBitsEq-1)

    # Remove sign bit from mantissa
    signMask = np.left_shift(1,nMantBits-1)
    onesMask = np.power(2,nMantBits)-1 # All ones
    codeMask = np.bitwise_xor(onesMask,signMask)

    code = np.bitwise_and(mantissa,codeMask) # Mantissa with no sign bit

    if scale == scaleMax:
        number = sign + code
    else:
        leadingOne = np.left_shift(1,nBitsEq-(scale+2)) # define leading 1
        mantShift = scaleMax-(scale+1) # amount to shift mantissa, +1 allows for leading one
        codeShift = np.left_shift(code,mantShift) # shift mantissa to the right of scale leading zeros

        # Add in trailing one after last mantissa bit, only if mantissa was shifted
        trailingOne = np.left_shift(1,mantShift-1)*(scale!=scaleMax-1)

        number = sign + leadingOne + codeShift + trailingOne # add it all together for final dequantized number

    aNum = DequantizeUniform(number,nBitsEq)
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits
    codeEq = QuantizeUniform(aNum,nBitsEq)

    signMask = np.left_shift(1,nBitsEq-1)
    signBit = np.right_shift(np.bitwise_and(signMask,codeEq),nBitsEq-1) # This gives value of sign bit
    sign = np.left_shift(signBit,nMantBits-1) # Stick sign bit in first position of mantissa

    code = np.bitwise_xor(np.bitwise_or(codeEq,signMask),signMask) # strip sign bit to get just |code|

    scaleMax = np.power(2,nScaleBits)-1 # Maximum value of scale bit

    if scale == scaleMax:
        mantissa = sign + code # Max leading 0's, Leave in leading 1
    else:
        shiftCode = np.right_shift(code,scaleMax-scale) # Bit shift larger code so it fits in mantissa
        mantissa = sign + shiftCode
    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    scaleMax = np.power(2,nScaleBits)-1
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits

    # Get sign bit from mantissa and put it in highest order bit
    signBit = np.right_shift(mantissa,nMantBits-1)
    sign = np.left_shift(signBit,nBitsEq-1)

    # Remove sign bit from mantissa
    signMask = np.left_shift(1,nMantBits-1)
    onesMask = np.power(2,nMantBits)-1 # All ones
    codeMask = np.bitwise_xor(onesMask,signMask)

    code = np.bitwise_and(mantissa,codeMask) # Mantissa with no sign bit

    if scale == scaleMax:
        number = sign + code
    else:
        mantShift = (scaleMax-(scale)) # amount to shift mantissa
        codeShift = np.left_shift(code,mantShift) # shift mantissa to the right of scale leading zeros

        # Add in trailing one after last mantissa bit, only if mantissa was shifted and is non-zero
        trailingOne = np.left_shift(1,mantShift-1)*(scale!=scaleMax-1)*(codeShift!=0)

        number = sign + codeShift + trailingOne # add it all together for final dequantized number

    aNum = DequantizeUniform(number,nBitsEq)
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits
    codeEq = vQuantizeUniform(aNumVec,nBitsEq)

    signMask = np.left_shift(np.ones_like(aNumVec,dtype=int),nBitsEq-1)
    signBit = np.right_shift(np.bitwise_and(signMask,codeEq),nBitsEq-1) # This gives value of sign bits
    sign = np.left_shift(signBit,nMantBits-1) # Stick sign bits in first position of mantissas

    code = np.bitwise_xor(np.bitwise_or(codeEq,signMask),signMask) # strip sign bit to get just |code|

    scaleMax = np.power(2,nScaleBits)-1 # Maximum value of scale bit
    shiftCode = np.right_shift(code,scaleMax-scale) # Bit shift larger code so it fits in mantissa

    mantissaVec = sign + shiftCode*(scale!=scaleMax) + code*(scale==scaleMax)
    ### YOUR CODE ENDS HERE ###

    return mantissaVec

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """
    ### YOUR CODE STARTS HERE ###
    scaleMax = np.power(2,nScaleBits)-1
    nBitsEq = np.power(2,nScaleBits)-1+nMantBits # Number of equivalent bits

    # Get sign bit from mantissa and put it in highest order bit
    signBit = np.right_shift(mantissaVec,nMantBits-1)
    sign = np.left_shift(signBit,nBitsEq-1)

    # Remove sign bit from mantissa
    signMask = np.left_shift(np.ones_like(mantissaVec,dtype=int),nMantBits-1)
    onesMask = np.ones_like(mantissaVec,dtype=int)*np.power(2,nMantBits)-1 # All ones
    codeMask = np.bitwise_xor(onesMask,signMask)

    code = np.bitwise_and(mantissaVec,codeMask) # Mantissa with no sign bit

    mantShift = (scaleMax-(scale)) # amount to shift mantissa
    codeShift = np.left_shift(code,mantShift) # shift mantissa to the right of scale leading zeros

    # Add in trailing one after last mantissa bit, only if mantissa was shifted and is non-zero
    trailingOne = np.left_shift(np.ones_like(mantissaVec,dtype=int),mantShift-1)*(scale!=scaleMax-1)*(codeShift!=0)

    number = sign + code*(scale==scaleMax) + (codeShift + trailingOne)*(scale!=scaleMax) # final dequantized number

    aNumVec = vDequantizeUniform(number,nBitsEq)
    ### YOUR CODE ENDS HERE ###

    return aNumVec

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    pass

    ### YOUR TESTING CODE ENDS HERE ###
