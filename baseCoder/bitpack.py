"""
bitpack_vector.py -- vectorized code for packing and unpacking bits into an array of bytes

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

from struct import *  # to pack and unpack arrays of bytes to/from strings
import numpy as np  # to get array capabilities
BYTESIZE=8

class PackedBits:
    """Object holding an array of bytes that one can read from/write to them as individual bits and which transfers the result in and out as a string"""

    def __init__(self):
        """Must be initialized with number of bytes to hold with a call to Size(nBytes) before starting to pack bits into it"""
        self.iByte=self.iBit =0 # starts w/ pointers at first bit

    def Size(self,nBytes):
        """Sizes an existing PackedBits object to hold nBytes of data (all initialized to zero)"""
        self.nBytes=nBytes # number of bytes it holds
        self.iByte=self.iBit=0 # reset pointers when resetting data
        self.data= np.zeros(nBytes,dtype=np.uint8)

    def GetPackedData(self):
        """Gets the packed data held by this PackedBits object and returns it as a data string"""
        s=self.data.tostring()
        return s

    def SetPackedData(self,data):
        """Sets the packed data held by this PackedBits object to the passed data string"""
        self.nBytes = len(data)
        self.data= np.fromstring(data,dtype=np.uint8)

    def WriteBits(self,info,nBits):
        """Writes lowest nBits of info into this PackedBits object at its current byte/bit pointers"""

        bitsLeft = nBits  # this is how many bits still need to get extracted from info

        # determine what can be fit in current byte
        nCur=BYTESIZE-self.iBit
        if nBits<nCur: nCur = nBits
        # compute mask to extract this from info
        infoMask = ((1<<nCur)-1)<<(bitsLeft-nCur)  # nCur ones starting at first bit to read
        # extract the bit pattern
        infoMask &= info  # now it has nCur bits from info starting at first bit to read
        # shift to right location for current byte
        if bitsLeft>BYTESIZE-self.iBit:
            dataMask = infoMask>>(bitsLeft-BYTESIZE+self.iBit)
        else:
            dataMask = infoMask<<(BYTESIZE-self.iBit-bitsLeft)
        # put it into data[iByte]
        self.data[self.iByte]+=dataMask
        # update count of remaining bits
        bitsLeft -= nCur
        if nCur+self.iBit==BYTESIZE:
            # we filled the byte
            self.iByte+=1
            self.iBit=0
        else:
            # didn't fill but done
            self.iBit += nCur  # didn't fill but done
            return

        # now do successive full bytes (if needed)
        nFull = bitsLeft//BYTESIZE  # this many full bytes
        for iFull in xrange(nFull):
            # compute mask to extract this from info
            infoMask = ((1<<BYTESIZE)-1)<<(bitsLeft-BYTESIZE)  # BYTESIZE ones starting at first bit to read
            # extract the bit pattern
            infoMask &= info  # now it has BYTESIZE bits from info starting at first bit to read
            # shift to right location for current byte
            if bitsLeft>BYTESIZE:
                dataMask = infoMask>>(bitsLeft-BYTESIZE)
            else:
                dataMask = infoMask<<(BYTESIZE-bitsLeft)
            # put it into data[iByte]
            self.data[self.iByte]+=dataMask
            # update count of remaining bits
            bitsLeft -= BYTESIZE
            self.iByte+=1

        # now do residual byte (if needed)
        if bitsLeft:
            # determine what can be fit in current byte
            nCur=BYTESIZE
            if bitsLeft<nCur: nCur = bitsLeft
            # compute mask to extract these last nCur bits from info
            infoMask = ((1<<nCur)-1) # rightmost nCur ones
            # extract the bit pattern
            infoMask &= info  # now it has last nCur bits from info
            # shift to right location for current byte
            if bitsLeft>BYTESIZE:
                dataMask = infoMask>>(bitsLeft-BYTESIZE)
            else:
                dataMask = infoMask<<(BYTESIZE-bitsLeft)
            # put it into data[iByte]
            self.data[self.iByte]+=dataMask
        # update bit pointer
        self.iBit=bitsLeft


    def ReadBits(self,nBits):
        """Returns next nBits of info from this PackedBits object starting at its current byte/bit pointers"""

        bitsLeft = nBits  # this is how many bits still need to get extracted from data
        info = 0  # this is where the bits will go

        # determine what can be extracted from current byte
        nCur = BYTESIZE-self.iBit  # this many bits to extract from current byte
        if nCur>nBits: nCur = nBits
        #compute mask to extract these bits from data
        dataMask = (((1<<nCur)-1)<<(BYTESIZE-self.iBit-nCur) )
        # extract the bit pattern
        dataMask &= self.data[self.iByte]  # now it has nCur bits from data starting at iBit
        # shift to right location for info
        if bitsLeft>BYTESIZE-self.iBit:
            infoMask = (dataMask<<(bitsLeft-BYTESIZE+self.iBit))
        else:
            infoMask= (dataMask>>(BYTESIZE-self.iBit-bitsLeft))
        # put it into info
        info+=infoMask
        # update count of remaining bits
        bitsLeft -= nCur
        if nCur+self.iBit==BYTESIZE:
            # we filled the byte
            self.iByte+=1
            self.iBit=0
        else:
            # didn't fill but done
            self.iBit += nCur  # didn't fill but done
            return info

        # now do successive full bytes (if needed)
        nFull = bitsLeft//BYTESIZE  # this many full bytes
        for iFull in xrange(nFull):
            # extract bit pattern of full byte
            dataMask = self.data[self.iByte]  # now it has nCur bits from data starting at iBit
             # shift to right location for info
            if bitsLeft>BYTESIZE-self.iBit:
                infoMask=dataMask<<(bitsLeft-BYTESIZE)
            else:
                infoMask=dataMask>>(BYTESIZE-bitsLeft)
            # put it into info
            info +=infoMask
            # update count of remaining bits
            bitsLeft -= BYTESIZE
            self.iByte+=1

        # now do residual byte (if needed)
        if bitsLeft:
            # determine what can be fit in current byte
            nCur = bitsLeft  # this many bits left to write from current byte
            # compute mask to extract these last nCur bits from data
            dataMask = ((1<<nCur)-1)<<(BYTESIZE-nCur) # leftmost nCur ones
            # extract the bit pattern
            dataMask &= self.data[self.iByte]  # now it has nCur bits from data starting at iBit
            # shift to right location for current byte
            if bitsLeft>BYTESIZE-self.iBit:
                infoMask=dataMask<<(bitsLeft-BYTESIZE)
            else:
                infoMask=dataMask>>(BYTESIZE-bitsLeft)
            # put it into info
            info += infoMask
        # update bit pointer
        self.iBit=bitsLeft

        # return info read out of data
        return info

    def ResetPointers(self):
        """Resets the pointers to the start of this PackedBits object (for example, to read out data that's been written in)"""
        self.iBit=self.iByte=0


#--------------------------------------------------------------------------------

# testing bit packing
if __name__=="__main__":

    print "\nTesting bit packing:"
    x = (3, 5, 11, 3, 1)
    xlen = (4,3,5,3,1)
    nBytes=2
    bp=PackedBits()
    bp.Size(nBytes)
    print "\nInput Data:\n",x ,"\nPacking Bit Sizes:\n",xlen
    for i in range(len(x)):
        bp.WriteBits(x[i],xlen[i])
    print "\nPacked Bits:\n",bp.GetPackedData()
    y=[]
    bp.ResetPointers()
    for i in range(len(x)):
        y.append(bp.ReadBits(xlen[i]))
    print "\nUnpacked Data:\n",y

    print "\n\nTesting file read/write for PackedBits objects:\n"
    print "\nPacked Bits:\n",bp.GetPackedData()
    from struct import pack,unpack
    fp = open("test.dat",'wb')
    s = bp.GetPackedData()
    fp.write(s)
    fp.close()
    del(bp)
    print "Bytes packed into a string to read/write:\t", s
    fp = open("test.dat",'rb')
    s=fp.read(nBytes)
    print "String read back from file:\t", s
    bp2=PackedBits()
    bp2.SetPackedData( s )
    print "\nRecovered Packed Bits in new PackedBits object:\n", bp2.GetPackedData()

    y=[]
    for i in range(len(x)):
        y.append(bp2.ReadBits(xlen[i]))
    print "\nUnpacked Data:\n",y



