"""
MS.py -- Module holding methods to implement M/S stereo coding

-----------------------------------------------------------------------
Copyright 2017 Ifueko Igbinedion -- All rights reserved
-----------------------------------------------------------------------

"""
import numpy as np

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


