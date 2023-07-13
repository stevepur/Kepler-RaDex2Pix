
import numpy as np
import types
import os
import array
from raDec2Pix import raDec2PixUtils as ru

_dataPath = os.path.join(os.path.dirname(__file__), "data")

class pointingSegment:
    def __init__(self):
        self.startMjd = 0;
        self.endMjd = 0;
        self.mjds = None;
        self.ras = None;
        self.declination = None;
        self.rolls = None;
    
class pointingModel:

    def __init__(self):
        self.pointingMjds = ru.make_col(np.loadtxt(_dataPath + "/pointingMjds.txt"))
        self.pointingRas = ru.make_col(np.loadtxt(_dataPath + "/pointingRas.txt"))
        self.pointingDecs = ru.make_col(np.loadtxt(_dataPath + "/pointingDeclinations.txt"))
        self.pointingRolls = ru.make_col(np.loadtxt(_dataPath + "/pointingRolls.txt"))
        self.pointingSegmentStartMjds = ru.make_col(np.loadtxt(_dataPath + "/pointingSegmentStartMjds.txt"))

    
    def get(self, fields):

        if type(fields) is not list:
            fields = [fields]
            
        returnData = None;
        for f in fields:
            if f == "mjds":
                returnData = ru.append_col(returnData, self.pointingMjds);
                
            if f == "ras":
                returnData = ru.append_col(returnData, self.pointingRas);
            
            if f == "declinations":
                returnData = ru.append_col(returnData, self.pointingDecs);
            
            if f == "rolls":
                returnData = ru.append_col(returnData, self.pointingRolls);
            
            if f == "segmentStartMjds":
                returnData = ru.append_col(returnData, self.pointingSegmentStartMjds);

        return returnData

    def generate_pointing_segment_array(self):
        # Generate a struct array of pointing models by pointing segment
        #

        # Get the pointing model fields
        mjds             = self.get( "mjds" );
        ras              = self.get( "ras" );
        declinations     = self.get( "declinations" );
        rolls            = self.get( "rolls" );
        segmentStartMjds = self.get( "segmentStartMjds" );
        
        # Intialize the output struct array
        startMjds = np.unique(segmentStartMjds);
        nSegments = startMjds.size;
        
        pointingSegments = [ pointingSegment() for i in range(nSegments) ]
        
        # Generate the pointing segments
        for iSegment in range(nSegments):

            isInSegment = segmentStartMjds == startMjds[iSegment];
            
            if iSegment > 0:
                pointingSegments[iSegment].startMjd = startMjds[iSegment];
            else:
                pointingSegments[iSegment].startMjd = mjds[0];

            pointingSegments[iSegment].mjds         = mjds[isInSegment];
            pointingSegments[iSegment].ras          = ras[isInSegment];
            pointingSegments[iSegment].declinations = declinations[isInSegment];
            pointingSegments[iSegment].rolls        = rolls[isInSegment];
            
        for iSegment in range(nSegments):
            if iSegment < nSegments-1:
                pointingSegments[iSegment].endMjd = pointingSegments[iSegment+1].startMjd;
            else:
                pointingSegments[iSegment].endMjd = pointingSegments[iSegment].mjds[-1];

        return pointingSegments
    
    def get_pointing(self, mjd):
    #
    # function pointing = get_pointing(pointingObject, mjd)
    #
    # Get the pointing (ra, dec, and roll) in degrees of the spacecraft
    # for the given input vector of MJDs.
    #
        from scipy import interpolate

        # column-vectorize the input mjd:
        #
        mjd = ru.make_col(np.array([mjd]));
        
        mjds = self.get("mjds");
        method = "cubic"; # default in matlab raDec2Pix
        
        if np.any(mjd < np.min(mjds)):
            raise ValueError("get_pointing: input mjd before valid range");

        if np.any(mjd > np.max(mjds)):
            raise ValueError("get_pointing: input mjd after valid range");

        # generate pointing segments array
        #
        pointingSegments = self.generate_pointing_segment_array();
        
        # define the return vectors
        #
        ra   = np.zeros(mjd.shape);
        dec  = np.zeros(mjd.shape);
        roll = np.zeros(mjd.shape);
        
        # loop over the pointing segments and get the pointing at the desired
        # mjd timestamps
        #
        nSegments = len(pointingSegments);
        
        for iSegment in range(nSegments):
            
            startMjd     = pointingSegments[iSegment].startMjd;
            endMjd       = pointingSegments[iSegment].endMjd;
            mjds         = pointingSegments[iSegment].mjds;
            ras          = pointingSegments[iSegment].ras;
            declinations = pointingSegments[iSegment].declinations;
            rolls        = pointingSegments[iSegment].rolls;
            
            if iSegment < nSegments-1:
                isInSegment = (mjd >= startMjd) & (mjd < endMjd);
            else:
                isInSegment = (mjd >= startMjd) & (mjd <= endMjd);
            
            if mjds.size==1:
                ra[isInSegment]   = ras;
                dec[isInSegment]  = declinations;
                roll[isInSegment] = rolls;
            else:
                interp1 = interpolate.interp1d(mjds, ras, kind=method);
                ra[isInSegment]   = interp1(mjd[isInSegment]);
                interp1 = interpolate.interp1d(mjds, declinations, kind=method);
                dec[isInSegment]  = interp1(mjd[isInSegment]);
                interp1 = interpolate.interp1d(mjds, rolls, kind=method);
                roll[isInSegment] = interp1(mjd[isInSegment]);
        
        pointing = [ra, dec, roll];
        
        return pointing
        
        
    

class rollTimeModel:

    def __init__(self):
        self.rollTimeMjds = ru.make_col(np.loadtxt(_dataPath + "/rollTimeMjds.txt"))
        self.rollTimeSeasons = ru.make_col(np.loadtxt(_dataPath + "/rollTimeSeasons.txt"))
        self.rollTimeRollOffsets = ru.make_col(np.loadtxt(_dataPath + "/rollTimeRollOffsets.txt"))
        self.rollTimeRas = ru.make_col(np.loadtxt(_dataPath + "/rollTimeFovCenterRas.txt"))
        self.rollTimeDecs = ru.make_col(np.loadtxt(_dataPath + "/rollTimeFovCenterDecs.txt"))
        self.rollTimeRolls = ru.make_col(np.loadtxt(_dataPath + "/rollTimeFovCenterRolls.txt"))

    def get_mjd_index(self, mjds):

        matchIndex = np.zeros(mjds.shape, dtype = int);
        for i in range(mjds.size):
            matchIndex[i] = np.argmax(self.rollTimeMjds > mjds[i]) - 1
        return matchIndex

    def get(self, julianTimes, fields):
        
        if type(fields) is not list:
            fields = [fields]
        
        julianTimes = ru.make_col(np.array(julianTimes));
        mjds = julianTimes - get_parameters("mjdOffset");
        matchIndex = self.get_mjd_index(mjds);
        
        returnData = None;
        for f in fields:
            if f == "mjds":
                returnData = ru.append_col(returnData, self.rollTimeMjds[matchIndex]);
                
            if f == "seasons":
                returnData = ru.append_col(returnData, self.rollTimeSeasons[matchIndex]);

            if f == "rollOffsets":
                returnData = ru.append_col(returnData, self.rollTimeRollOffsets[matchIndex]);
            
            if f == "rollTimeRas":
                returnData = ru.append_col(returnData, self.rollTimeRas[matchIndex]);
            
            if f == "rollTimeDecs":
                returnData = ru.append_col(returnData, self.rollTimeDecs[matchIndex]);
            
            if f == "rollTimeRolls":
                returnData = ru.append_col(returnData, self.rollTimeRolls[matchIndex]);

        return returnData

    def juliandate2quarter(self, julianTimes):
        return self.get(julianTimes, "seasons")
        

class geometryModel:
    # data tables
    geometryConstants = -1;
    geometryUncertainties = -1;
    
    def __init__(self):
        self.geometryConstants = ru.make_col(np.loadtxt(_dataPath + "/geometryConstants.txt"))
        self.geometryUncertainties = ru.make_col(np.loadtxt(_dataPath + "/geometryUncertainty.txt"))

    def get(self, fields):
        
        if type(fields) is not list:
            fields = [fields]
            
        chipTransStart = 0;
        nChipTrans = 126;
        chipOffsetStart = 126;
        nChipOffset = 126;
        plateScaleStart = 252;
        nPlateScale = 84;
        pincushionStart = 252 + 84;
        nPincushion = 84;

        returnData = None;
        for f in fields:
            if f == "constants":
                returnData = ru.append_col(returnData, self.geometryConstants);

            if f == "uncertainties":
                returnData = ru.append_col(returnData, self.geometryUncertainties);

            if f == "chipTrans":
                returnData = ru.append_col(returnData, self.geometryConstants[chipTransStart:chipTransStart+nChipTrans]);
            
            if f == "chipOffset":
                returnData = ru.append_col(returnData, self.geometryConstants[chipOffsetStart:chipOffsetStart+nChipOffset]);
            
            if f == "plateScale":
                returnData = ru.append_col(returnData, self.geometryConstants[plateScaleStart:plateScaleStart+nPlateScale]);
            
            if f == "pincushion":
                returnData = ru.append_col(returnData, self.geometryConstants[pincushionStart:pincushionStart+nPincushion]);

        return returnData
        
        
        
        
        

def get_parameters(field):
    if field == "nominalClockingAngle":
        return 13
        
    if field == "halfOffsetModuleAngleDegrees":
        return 1.43
        
    if field == "outputsPerRow":
        return 10
        
    if field == "outputsPerColumn":
        return 10
    
    if field == "nRowsImaging":
        return 1024

    if field == "nColsImaging":
        return 1100
        
    if field == "nMaskedSmear":
        return 20

    if field == "nLeadingBlack":
        return 12

    if field == "nModules":
        return 21
        
    if field == "mjdOffset":
        return 2.400000500000000e+06

def get_channel_numbers():
    return np.array([
        [ 0,  0, 56, 55, 36, 35, 16, 15,  0,  0],
        [ 0,  0, 53, 54, 33, 34, 13, 14,  0,  0],
        [75, 74, 60, 59, 40, 39, 17, 20,  1,  4],
        [76, 73, 57, 58, 37, 38, 18, 19,  2,  3],     # note: layout of matrix mimics
        [79, 78, 63, 62, 44, 43, 21, 24,  5,  8],     # actual physical layout of chn numbers
        [80, 77, 64, 61, 41, 42, 22, 23,  6,  7],     # on FPA
        [83, 82, 67, 66, 46, 45, 26, 25,  9, 12],
        [84, 81, 68, 65, 47, 48, 27, 28, 10, 11],
        [ 0,  0, 70, 69, 50, 49, 30, 29,  0,  0],
        [ 0,  0, 71, 72, 51, 52, 31, 32,  0,  0]])
