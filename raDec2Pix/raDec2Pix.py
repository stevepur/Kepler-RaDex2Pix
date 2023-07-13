
import numpy as np
from raDec2Pix import raDec2PixUtils as ru
from raDec2Pix import raDec2PixModels as rm
from raDec2Pix import raDec2PixAberrate

class raDec2PixClass:
    def __init__(self):
        
        self.aberrate = raDec2PixAberrate.aberrateRaDec()
        self.geomModel = rm.geometryModel()
        self.rollTimeModel = rm.rollTimeModel()
        self.pointingModel = rm.pointingModel()
    
    def get_direction_matrix(self, julianTime, seasonInt, drot1, rot2, rot3):
        drot1 = np.array(drot1);
        rot2 = np.array(rot2);
        rot3 = np.array(rot3);

        # Determine the season and roll offset based on the julianTime.
        rollTimeData = self.rollTimeModel.get(julianTime,["rollOffsets","seasons"]);
        rollOffset = rollTimeData[:, 0];
        seasonInt = rollTimeData[:, 1];

        deg2rad = np.pi/180.0;

        # Calculate the Direction Cosine Matrix to transform from RA and Dec to FPA coordinates
        rot1 = rm.get_parameters('nominalClockingAngle') + rollOffset + seasonInt * 90.0;
        rot1 = rot1 + drot1; # add optional offset in X'-axis rotation
        rot1 = rot1 + 180; # Need to account for 180 deg rotation of field due to imaging of mirror
        rot1 = np.fmod(rot1, 360); # make small if rot1 < -360 or rot1 > 360

        if (rot1.size != rot2.size)  | (rot1.size != rot3.size):
            rot2 = np.tile(rot2,rot1.shape);
            rot3 = np.tile(rot3,rot1.shape);
            
        rot1 = ru.make_col(rot1)
        rot2 = ru.make_col(rot2)
        rot3 = ru.make_col(rot3)

        srac  = np.sin(rot3*deg2rad); # sin phi 3 rotation
        crac  = np.cos(rot3*deg2rad); # cos phi
        sdec  = np.sin(rot2*deg2rad); # sin theta 2 rotation Note 2 rotation is negative of dec in right hand sense
        cdec  = np.cos(rot2*deg2rad); # cos theta
        srotc = np.sin(rot1*deg2rad); # sin psi 1 rotation
        crotc = np.cos(rot1*deg2rad); # cos psi
        
        # Extract the focal plane geometry constants
        geometryData = self.geomModel.get(["chipTrans","chipOffset"]);
        chipTrans = np.transpose(np.reshape(geometryData[:, 0], (3, 42), order='F').copy());
        chipOffset = np.transpose(np.reshape(geometryData[:, 1], (3, 42), order='F').copy());

        #     DCM for a 3-2-1 rotation, Wertz p764
        DCM11 =  cdec*crac;
        DCM12 =  cdec*srac;
        DCM13 = -sdec;
        DCM21 = -crotc*srac+srotc*sdec*crac;
        DCM22 =  crotc*crac+srotc*sdec*srac;
        DCM23 =  srotc*cdec;
        DCM31 =  srotc*srac+crotc*sdec*crac;
        DCM32 = -srotc*crac+crotc*sdec*srac;
        DCM33 =  crotc*cdec;
        
        #     Calculate DCM for each chip relative to center of FOV
        nModules2 = rm.get_parameters('nModules') * 2;
        DCM11c      = np.zeros((nModules2,1));
        DCM12c      = np.zeros((nModules2,1));
        DCM13c      = np.zeros((nModules2,1));
        DCM21c      = np.zeros((nModules2,1));
        DCM22c      = np.zeros((nModules2,1));
        DCM23c      = np.zeros((nModules2,1));
        DCM31c      = np.zeros((nModules2,1));
        DCM32c      = np.zeros((nModules2,1));
        DCM33c      = np.zeros((nModules2,1));
        
    #    print(chipTrans)
        for i in range(nModules2): # step through each chip
            srac  = np.sin(deg2rad*chipTrans[i,0]); # sin phi 3 rotation
            crac  = np.cos(deg2rad*chipTrans[i,0]); # cos phi
            sdec  = np.sin(deg2rad*chipTrans[i,1]); # sin theta 2 rotation
            cdec  = np.cos(deg2rad*chipTrans[i,1]); # cos theta
            srotc = np.sin(deg2rad*(chipTrans[i,2]+chipOffset[i,0])); # sin psi 1 rotation includes rotation offset
            crotc = np.cos(deg2rad*(chipTrans[i,2]+chipOffset[i,0])); # cos psi

            # DCM for a 3-2-1 rotation, Wertz p762
            DCM11c[i,0] = cdec*crac;
            DCM12c[i,0] = cdec*srac;
            DCM13c[i,0] =-sdec;
            DCM21c[i,0] =-crotc*srac + srotc*sdec*crac;
            DCM22c[i,0] = crotc*crac + srotc*sdec*srac;
            DCM23c[i,0] = srotc*cdec;
            DCM31c[i,0] = srotc*srac + crotc*sdec*crac;
            DCM32c[i,0] =-srotc*crac + crotc*sdec*srac;
            DCM33c[i,0] = crotc*cdec;
        
        return DCM11,  DCM12,  DCM13,  DCM21,  DCM22,  DCM23,  DCM31,  DCM32,  DCM33, DCM11c, DCM12c, DCM13c, DCM21c, DCM22c, DCM23c, DCM31c, DCM32c, DCM33c, chipOffset;

    def process_in_FOV(self, inFov, chnI, chnJ, chnNum, DCM11c, DCM12c, DCM13c, DCM21c, DCM22c, DCM23c, DCM31c, DCM32c, DCM33c, lpf, mpf, npf, chipOffset):

        chnI = np.array(chnI, dtype = int);
        chnJ = np.array(chnJ, dtype = int);

        lpf = np.reshape(lpf, (chnI.size, 1));
        mpf = np.reshape(mpf, (chnI.size, 1));
        npf = np.reshape(npf, (chnI.size, 1));

        # Inernal function to perform the transformation calc into chipspace.
        #
        
        chnN = np.zeros(chnI.shape)
        chnN[inFov, 0] = chnNum[chnI[inFov, 0], chnJ[inFov, 0]];
        chnN = np.reshape([chnN], (chnI.size, 1));

        chipN = np.zeros(chnI.shape, dtype=int)
        chipN[inFov, 0] = np.fix((chnN[inFov, 0]+1)/2).astype(int)-1;  # chip number index,

        # set up temporary variables to speed the transform lines DAC 16 Mar 2005
        lpFOV    =    lpf[inFov, 0];
        mpFOV    =    mpf[inFov, 0];
        npFOV    =    npf[inFov, 0];
        chipNFov = chipN[inFov, 0];
        lpFOV = np.reshape(lpFOV, (lpFOV.size, 1));
        mpFOV = np.reshape(mpFOV, (lpFOV.size, 1));
        npFOV = np.reshape(npFOV, (npFOV.size, 1));
        inFov = np.reshape(inFov, (inFov.size, 1));

        
        lpm = DCM11c[chipNFov]  * lpFOV + DCM12c[chipNFov]  * mpFOV + DCM13c[chipNFov]  * npFOV;
        mpm = DCM21c[chipNFov]  * lpFOV + DCM22c[chipNFov]  * mpFOV + DCM23c[chipNFov]  * npFOV;
        npm = DCM31c[chipNFov]  * lpFOV + DCM32c[chipNFov]  * mpFOV + DCM33c[chipNFov]  * npFOV;


        # Define chip coor as: rotation about the center of the module(field flattener lens) &
        #     angular row and column from this center then column 1100 is angle zero
        #     and decreases up and down with increasing angle towards readout amp on
        #     each corner and row 1024 starts after a gap of 39 pixels decreasing
        #     with increasing angle
        #
        rad2deg = 180/np.pi;
        latm =np.arcsin(npm);                      # transformed lat +Z' to chip coor in radians
        lngm =rad2deg*np.arctan2(mpm,lpm) * 3600.;   # transformed long +Y' to chip coor in arc sec
        lngr =lngm*np.cos(latm);                # correct for cos effect going from spherical to rectangular coor
                                              # one deg of long at one deg of lat is smaller than one deg at zero lat
                                              # by cos(lat) , amounts to 1/2 arc sec=1/8 pix
        latm =rad2deg*latm*3600;   # latm in arc sec

        # Now convert to row and column
        #

        #   obtain the plate scales and generate a vector of plate scales which goes with the
        #   vector of latitude / longitude coordinates.  Note that we are using 1 plate scale per
        #   CCD here -- the "even-numbered" mod/out's plate scale is used for both even and odd
        #   mod/outs.
            
        plateScalesAll = self.geomModel.get("plateScale");
        plateScaleN = np.reshape(plateScalesAll[(chipNFov)*2], lngr.shape);

        #   obtain the pincushion parameters and reshape them along the same lines as the plate
        #   scales
        #   there is only one geometry model.
            
        pincushionAll = self.geomModel.get("pincushion");
        pincushionN = np.reshape(pincushionAll[(chipNFov)*2], lngr.shape) ;
        
        chipOffsetWork2 = np.reshape(chipOffset[chipNFov,1], lngr.shape);
        chipOffsetWork3 = np.reshape(chipOffset[chipNFov,2], lngr.shape);

        #   convert the longitude and latitude from arcsec to pixels, and take into account the
        #   pincushion aberration -- these are equivalent to row and column, but with an origin of
        #   coordinates at the notional center of the module

        radius2 = lngr**2 + latm**2 ;
        rowCentered = lngr / plateScaleN * ( 1 + pincushionN * radius2 ) ;
        colCentered = latm / plateScaleN * ( 1 + pincushionN * radius2 ) ;

        #   apply fixed offsets to get to the correct origin of coordinates
        row = np.zeros(chnI.shape)
        column = np.zeros(chnI.shape)
        module = np.zeros(chnI.shape)
        output = np.zeros(chnI.shape)

        pRow = rm.get_parameters("nRowsImaging") - rowCentered + chipOffsetWork2 ;
        row[inFov, 0] = pRow;

        colRecentered = colCentered - chipOffsetWork3 ;

        pColn = np.zeros(colRecentered.shape);
        # positive side of chip
        gez = np.where(colRecentered >= 0.0)[0];
        pColn[gez] = rm.get_parameters("nColsImaging") - colRecentered[gez];
        # bottom half of chip
        chnN[inFov[gez],0] = np.fix((chnN[inFov[gez],0] - 1) / 2) * 2 + 1;

        # negative side of chip
        lz = np.where(colRecentered < 0.0)[0];
        pColn[lz] = rm.get_parameters("nColsImaging") + colRecentered[lz];

        # top half of chip
        chnN[inFov[lz],0] = np.fix((chnN[inFov[lz],0] + 1)/2) * 2;

        # set column & output values
        column[inFov, 0] = pColn;
        output[inFov, 0] = chnN[inFov, 0];

        # determine module number
        mtemp = np.ceil(output[inFov, 0]/4)+1;

        mtemp[mtemp >=  5] = mtemp[mtemp >=  5] + 1; # add one to account for missing module 5
        mtemp[mtemp >= 21] = mtemp[mtemp >= 21] + 1; # add one again to account for missing module 21
        module[inFov, 0] = mtemp;

        # Return the nonzero calculated data:
        #
        nonZeroIndex = module != 0;
        module = module[nonZeroIndex];
        output = output[nonZeroIndex];
        row    = row[   nonZeroIndex];
        column = column[nonZeroIndex];

        return module, output, row, column

    def do_transform(self, ra, dec, quarter, julianTime, raPointing, decPointing, rollPointing):

        ra = ru.make_col(np.array(ra))
        dec = ru.make_col(np.array(dec))
        julianTime = ru.make_col(np.array(julianTime))
        raPointing = ru.make_col(np.array(raPointing))
        decPointing = ru.make_col(np.array(decPointing))
        rollPointing = ru.make_col(np.array(rollPointing))

        chnNum = rm.get_channel_numbers();

        ra0    = raPointing;    # use input FOV right ascension
        dec0   = decPointing;    # use input FOV dec
        drot1  = rollPointing;  # use input FOV rotation offset

        # Initialize output arrays
        #
        module = np.zeros(ra.shape);
        output = np.zeros(ra.shape);
        row    = np.zeros(ra.shape);
        column = np.zeros(ra.shape);

        rot3 = ra0;          # 3-rot corresponds to FOV at nominal RA.
        rot2 = -dec0;        # 2-rot corresponds to FOV at nomial Dec. Note minus sign.
        #    since +dec corresponds to -rot


        # Get the direction cosine matrix and chip geometry constants for the
        # input julian times.
        #
        DCM11,  DCM12,  DCM13,  DCM21,  DCM22,  DCM23,  DCM31,  DCM32,  DCM33, DCM11c, DCM12c, DCM13c, DCM21c, DCM22c, DCM23c, DCM31c, DCM32c, DCM33c, chipOffset = self.get_direction_matrix(julianTime, quarter, drot1, rot2, rot3)
        
        deg2rad = np.pi/180.0;
        rad2deg = 180/np.pi;
        trar  = deg2rad*ra; # target right ascension (radians) optionally array of values
        tdecr = deg2rad*dec; # target declination (radians) optionally array of values

        cosa = np.cos(trar) * np.cos(tdecr);
        cosb = np.sin(trar) * np.cos(tdecr);
        cosg = np.sin(tdecr);


        # Now do coordinate transformation: get direction cosines in FPA coordinates
        #
        lpf = DCM11 * cosa + DCM12 * cosb + DCM13 * cosg;
        mpf = DCM21 * cosa + DCM22 * cosb + DCM23 * cosg;
        npf = DCM31 * cosa + DCM32 * cosb + DCM33 * cosg;

        # Convert dir cosines to longitude and lat in FPA coor system
        #
        lat = rad2deg*np.arcsin( npf);    # transformed  lat +Z' in deg
        lng = rad2deg*np.arctan2(mpf,lpf); # transformed long +Y' in deg

        # find which channel this falls onto (+5 to center on the 10-output grid,
        # +1 for 1-offset matlab arrays).  The chnI and chnJ vars are the row and
        # column indices into this 10x10 grid.
        #
        chnI=np.floor(lat/rm.get_parameters("halfOffsetModuleAngleDegrees")) + 5;
        chnJ=np.floor(lng/rm.get_parameters("halfOffsetModuleAngleDegrees")) + 5;
        chnI = np.reshape(chnI, (chnI.size, 1));
        chnJ = np.reshape(chnJ, (chnI.size, 1));

        # Find inputs that are outside the FOV
        #
        outOfFov = np.where((chnI<0) | (chnI>9) | (chnJ<0) | (chnJ>9)  # off FOV
                            | ((chnI<=1) & (chnJ<=1))  # exclude module 1
                            | ((chnI<=1) & (chnJ>=8))  # exclude module 5
                            | ((chnI>=8) & (chnJ<=1))  # exclude module 21
                            | ((chnI>=8) & (chnJ>=8)));   # exclude module 25
        outOfFov = outOfFov[0];

        offFovCode = -1;

        row = np.zeros(chnI.shape)
        column = np.zeros(chnI.shape)
        module = np.zeros(chnI.shape)
        output = np.zeros(chnI.shape)
        
        module[outOfFov,0] = offFovCode;
        output[outOfFov,0] = offFovCode;
        row[outOfFov,0]    = offFovCode;
        column[outOfFov,0] = offFovCode;
        
        inFov = np.where(module != offFovCode);  # In FOV means it hasn't been set to "out" yet.
        inFov = inFov[0];
        
        if inFov.size > 0:
            fovDat = self.process_in_FOV(inFov, chnI, chnJ, chnNum, DCM11c, DCM12c, DCM13c, DCM21c, DCM22c, DCM23c, DCM31c, DCM32c, DCM33c, lpf, mpf, npf, chipOffset);
            module[inFov] = fovDat[0].reshape(inFov.size,1);
            output[inFov] = fovDat[1].reshape(inFov.size,1);
            row[   inFov] = fovDat[2].reshape(inFov.size,1);
            column[inFov] = fovDat[3].reshape(inFov.size,1);

        output = 1 + np.mod(output-1,4);

        return module, output, row, column;

    def calcPixFromRaDec(self, ra, dec, quarter, julianTime, raPointing, decPointing, rollPointing):
        chunkSize = 100000;
        ra = np.array(ra);
        dec = np.array(dec);
        quarter = np.array(quarter);
        julianTime = np.array(julianTime);
        raPointing = np.array(raPointing);
        decPointing = np.array(decPointing);
        rollPointing = np.array(rollPointing);
        
        if ra.size < chunkSize:
            module, output, row, column = self.do_transform(ra, dec, quarter, julianTime, raPointing, decPointing, rollPointing)
        else:
            module = np.zeros((ra.size, 1));
            output = np.zeros((ra.size, 1));
            row    = np.zeros((ra.size, 1));
            column = np.zeros((ra.size, 1));
            
            computing = True;
            dStart = 0;
            while computing:
                dEnd = dStart + chunkSize-1;
                if dEnd >= ra.size:
                    dEnd = ra.size-1;
                    computing = False;
                dRange = np.arange(dStart,dEnd+1, dtype=int)
                dRange = np.reshape(dRange, (dEnd-dStart+1, 1))
                m, o, r, c = self.do_transform(ra[dRange], dec[dRange], quarter[dRange], julianTime[dRange], raPointing[dRange], decPointing[dRange], rollPointing[dRange]);
                module[dRange, 0] = m;
                output[dRange, 0] = o;
                row[dRange, 0] = r;
                column[dRange, 0] = c;
                
                dStart = dStart + chunkSize;
                if dStart >= ra.size:
                    computing = False;

        return module, output, row, column



    def RaDec2Pix(self, ra, dec, julianTime, raPointing, decPointing,  rollPointing, aberrateFlag=True):
    #
    # function [module, output, row, column] = RaDec2Pix(raDec2PixObject, ra, dec, julianTime, raPointing, decPointing,  rollPointing, [aberrateFlag])
    #
    # INPUTS:
    #    ra         - the right ascension of the star(s), in degrees.  If there is more than one
    #                 star, this argument should be a vector.  The ra and dec arguments
    #                 must be the same size.
    #    dec        - the declination of the star(s), in degrees.  If there is more than one
    #                 star, this argument should be a vector. The ra and dec arguments
    #                 must be the same size.
    #    julianTime - the julian times to do the (RA,Dec)->Pix conversion
    #                 for.  The size of this argument must be:
    #                     1) the same as ra and dec,
    #                     2) a single value, or
    #                     3) an arbitrary length vector, iff ra and dec are
    #                        single values
    #    raPointing    - Optional argument specifying focal plane center in RA, in
    #            degrees.  raPointing, decPointing, and rollPointing must all be specified, or none.
    #    decPointing    - Optional argument specifying focal plane center in declination, in degrees.
    #    rollPointing  - Optional argument specifying focal plane rotation, in degrees.
    #
    #        If (raPointing, decPointing, rollPointing) focal plane pointing is not specified, the nominal
    #        pointing is used.
    #
    #        If specified, the (raPointing, decPointing, rollPointing) args must be the same length as julianTime.
    #
    #    aberrateFlag - Optional argument to turn aberration on or off. The default is on.
    #            If raPointing, decPointing, and phiPoinnting are not specified, this can be used as the
    #            fourth argument.
    #
    #    If the aberrateFlag flag is on (the default is on), the input RAs and Decs
    #      are assumed to be sky coordinates unaffected by the aberration of
    #      starlight.
    #
    #
    #
    # OUTPUTS:
    #    [module output row column] Four column vectors of the same size as the
    #    largest of the input arguments (ra, dec, and julianTime)
    #
    #
    # SAMPLE USE CASES:
    #
    #    The simplest case:  single-element inputs and outputs:
    #        ra = 300, dec = 45, jt = 2455000, [module output row col] = RaDec2Pix(ra, dec, jt);
    #
    #    Multiple element ra/dec inputs, single julianTime inputs.  The outputs will have the same size as the
    #    ra/dec inputs:
    #        ra = 300.0:.01:300.1, dec = 45.0:.01:45.1, jt = 2455000, [module output row col] = RaDec2Pix(ra, dec, jt);
    #
    #    Single element ra/dec inputs, multiple-element julianTime inputs.  The outputs will have the same size as the
    #    time input:
    #        ra = 300, dec = 45, jt = 2455000:1:2455100, [module output row col] = RaDec2Pix(ra, dec, jt);
    #
    #    Multiple element ra/dec inputs, multiple (same size as ra/dec) julianTime inputs.  The outputs will have the same size as the
    #    ra/dec/time inputs:
    #        ra = 300.0:.01:300.1, dec = 45.0:.01:45.1, jt = 2455000:.01:2455000.1, [module output row col] = RaDec2Pix(ra, dec, jt);
    #
     
        
        ra = ru.make_col(np.array(ra))
        dec = ru.make_col(np.array(dec))
        julianTime = ru.make_col(np.array(julianTime))
        raPointing = ru.make_col(np.array(raPointing))
        decPointing = ru.make_col(np.array(decPointing))
        rollPointing = ru.make_col(np.array(rollPointing))    # input checking

        if aberrateFlag:
            abDat = self.aberrate.aberrate_ra_dec(ra, dec, julianTime);
            ra = abDat[0];
            dec = abDat[1];
            
        if np.any(np.isnan(ra)) | np.any(np.isnan(dec)) | np.any(np.isnan(julianTime)):
            raise ValueError("RaDec2Pix: nan in ra, dec or julianTime");
            
        if ra.size != dec.size:
            raise ValueError("RaDec2Pix: ra and dec are not the same size");
            
        # Make (ra,dec) and julianTime the same size if one is different from
        # the other:
        #
        
        if (julianTime.size == 1) & (ra.size > 1):
            julianTime = np.tile(julianTime, ra.shape);

        if (ra.size == 1) & (julianTime.size > 1):
            ra  = np.tile(ra, julianTime.shape);
            dec = np.tile(dec, julianTime.shape);

        # check that the inputs are in a legal range
        # 2.45e6 (Oct 1995) and 2.46e6 (Feb 2023) are chosen as conservative julian
        # date bounds to avoid passing ridiculous values to the SPICE library call
        # (for the spacecraft state vector).
        if (np.min(ra) < 0.) | (np.max(ra) > 360.) \
            | (np.min(dec) < -90.) | (np.max(dec) > 90.) \
            | (np.min(julianTime) < 2.45e6) | (np.max(julianTime) > 2.46e6):
            raise ValueError("RaDec2Pix: ra, dec or julianTime have illegal values");
            
        # Verify pointing arguments are consistent lengths:
        if np.unique([raPointing.size, decPointing.size, rollPointing.size]).size > 1:
            raise ValueError("RaDec2Pix: raPointing, decPointing, and rollPointing are not the same size");

        quarter = self.rollTimeModel.juliandate2quarter(julianTime);
            
        # If raPointing, decPointing, and rollPointing came in as 1x1 values, stretch them to the size of
        # the other input arguments for the unique rows operation later on.
        if (raPointing.size == 1) & (ra.size > 1):
            raPointing = np.tile(raPointing, ra.shape);
            decPointing = np.tile(decPointing, ra.shape);
            rollPointing = np.tile(rollPointing, ra.shape);

        # Check to see if season, raPointing, decPointing, and rollPointing are constant
        #   if they are, then calcPixFromRaDec can be called all at once.
        #
        pointingIsConstant = (np.unique(raPointing).size == 1) & \
                             (np.unique(decPointing).size == 1) & \
                             (np.unique(rollPointing).size == 1) & \
                             (np.unique(quarter).size == 1);

        # Perform the actual geometric transformation:
        #
#        if (np.unique(julianTime).size == 1) & pointingIsConstant:
        if (np.unique(julianTime).size == 1) & pointingIsConstant:
            module, output, row, column = self.calcPixFromRaDec(ra, dec, quarter, julianTime, raPointing, decPointing, rollPointing);
        else:
            module = np.zeros((julianTime.size, 1));
            output = np.zeros((julianTime.size, 1));
            row    = np.zeros((julianTime.size, 1));
            column = np.zeros((julianTime.size, 1));

            for it in range(julianTime.size):
                module[it], output[it], row[it], column[it] = self.calcPixFromRaDec(ra[it], dec[it], quarter[it], julianTime[it], raPointing[it], decPointing[it], rollPointing[it]);
            
        return [module, output, row, column]




    def ra_dec_2_pix(self, ra, dec, mjds,
                raPointing = None, decPointing = None, rollPointing = None,
                deltaRa = None, deltaDec = None, deltaRoll = None, aberrateFlag = True):

        ra = ru.make_col(np.array([ra]))
        dec = ru.make_col(np.array([dec]))
        mjds = ru.make_col(np.array([mjds]))
        isReturnMatrix = (ra.size > 1) & (mjds.size > 1);

        module = np.zeros((ra.size, mjds.size));
        output = np.zeros((ra.size, mjds.size));
        row    = np.zeros((ra.size, mjds.size));
        column = np.zeros((ra.size, mjds.size));

        jds = mjds + rm.get_parameters("mjdOffset");

        if (np.all(raPointing == None)) | (np.all(raPointing == -1)):
            pointing = self.pointingModel.get_pointing(mjds);
            raPointing = pointing[0];
            decPointing = pointing[1];
            rollPointing = pointing[2];
        else:
            raPointing = ru.make_col(np.array([raPointing]))
            decPointing = ru.make_col(np.array([decPointing]))
            rollPointing = ru.make_col(np.array([rollPointing]))
            
        if np.any(deltaRa != None):
            raPointing = raPointing + ru.make_col(np.array([deltaRa]));
            decPointing = decPointing + ru.make_col(np.array([deltaDec]));
            rollPointing = rollPointing + ru.make_col(np.array([deltaRoll]));


        if isReturnMatrix:
            for itime in range(jds.size):
                pixData = self.RaDec2Pix(ra, dec, jds[itime], raPointing[itime], decPointing[itime], rollPointing[itime], aberrateFlag);
                module[:,itime] = pixData[0].ravel();
                output[:,itime] = pixData[1].ravel();
                row[:,itime] = pixData[2].ravel();
                column[:,itime] = pixData[3].ravel();
        else:
            pixData = self.RaDec2Pix(ra, dec, jds, raPointing, decPointing, rollPointing, aberrateFlag);
            module = pixData[0];
            output = pixData[1];
            row = pixData[2];
            column = pixData[3];

        # Adjust inputs to move center of pixel to (0.0, 0.0);
        #
        row = row - 0.5;
        column = column - 0.5;

        # RaDec2Pix gives row/col on the visable silicon.  Adjust the outputs to be on total accumulation memory:
        #
        row    = row    + rm.get_parameters("nMaskedSmear");
        column = column + rm.get_parameters("nLeadingBlack");
        
#        return module, output, row, column
        if module.size == 1:
            return module[0,0].astype(int), output[0,0].astype(int), row[0,0], column[0,0]
        elif module.shape[1] == 1:
            return module[:,0].astype(int), output[:,0].astype(int), row[:,0], column[:,0]
        else:
            return module.astype(int), output.astype(int), row, column
      
      

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
    #=========================================================================================
    
    # subfunction which iteratively computes the latitude and longitude from the row and
    # column, given the plate scale and pincushion parameters
    
    def compute_lat_long_iteratively(self, rowCentered, colCentered, plateScaleN, pincushionN, convergenceTolerance ):
   
    # define a vector which shows the convergence status of each member of the data vectors,
    # and initialize to false
        convergenceStatus = np.zeros(rowCentered.shape, dtype=bool); 
      
    # initialize lngr and latm to the values they would have for no pincushion correction
    
        lngr = rowCentered * plateScaleN ;
        latm = colCentered * plateScaleN ;
        radius2 = np.zeros(lngr.shape) ;
        lngrNew = np.zeros(lngr.shape) ;
        latmNew = np.zeros(lngr.shape) ;
      
    # iterate as long as there are members which are not yet converged
    
        iconverge = 0;
        converganceAttemptsLimit = 100;
        while ( (not np.all(convergenceStatus)) & (iconverge < converganceAttemptsLimit)):
            iconverge = iconverge + 1;
          
    #     use the old values of lngr and latm to compute new ones, in the case where
    #     convergence hasn't happened yet
    
            needToIterate = np.where(~convergenceStatus) ;
            radius2[needToIterate] = lngr[needToIterate]**2 + latm[needToIterate]**2 ;
            lngrNew[needToIterate] = rowCentered[needToIterate] * plateScaleN[needToIterate] / ( 1 + pincushionN[needToIterate] * radius2[needToIterate] ) ;
            latmNew[needToIterate] = colCentered[needToIterate] * plateScaleN[needToIterate] / ( 1 + pincushionN[needToIterate] * radius2[needToIterate] ) ;
            deltaLngr = np.abs(lngrNew - lngr) ;
            deltaLatm = np.abs(latmNew - latm) ;
         
    #     Convergence occurs when either the absolute change is < the convergenceTolerance or
    #     else the relative change is < the convergence tolerance.  We expect the former
    #     condition to hold when the value of lngr / latm is close to zero, otherwise the
    #     latter.  Since I don't like hugely complicated logical expressions, I'll do this one
    #     in pieces
    
            absoluteConvergenceLngr = deltaLngr < convergenceTolerance ;
            absoluteConvergenceLatm = deltaLatm < convergenceTolerance ;
            relativeConvergenceLngr = deltaLngr < convergenceTolerance * np.abs(np.max([[lngrNew], [lngr]])) ;
            relativeConvergenceLatm = deltaLatm < convergenceTolerance * np.abs(np.max([[latmNew], [latm]])) ;
              
    #     build an overall convergence vector from the 4 component vectors
    
            convergenceLngr = absoluteConvergenceLngr | relativeConvergenceLngr ;
            convergenceLatm = absoluteConvergenceLatm | relativeConvergenceLatm ;
            convergenceStatus = convergenceLngr & convergenceLatm ;
          
    #     replace the old estimated result vectors with the new ones
    
            lngr[:] = lngrNew[:] ;
            latm[:] = latmNew[:] ;
          
    
        if iconverge > converganceAttemptsLimit:
            print("raDec2Pix:Pix2RaDec: compute_lat_long_iteratively did not converge.");
        return lngr, latm
        
    def doTransform_pix(self, module, output, row, column, quarter, julianTime, raPointing, decPointing, rollPointing):
    
        module = ru.make_col(np.array(module))
        output = ru.make_col(np.array(output))
        row = ru.make_col(np.array(row))
        column = ru.make_col(np.array(column))
        julianTime = ru.make_col(np.array(julianTime))
        raPointing = ru.make_col(np.array(raPointing))
        decPointing = ru.make_col(np.array(decPointing))
        rollPointing = ru.make_col(np.array(rollPointing))

        module = module.astype(int);
        output = output.astype(int);
        
        rtd = 180/np.pi;


#        new_mod = [-1, 0:2, -1, 3:17, -1, 18:20, -1];
        new_mod = np.r_[-1, 0:3, -1, 3:18, -1, 18:21, -1];
        output  = output + 4*new_mod[module-1];

        ra0    = raPointing;
        dec0   = decPointing;
        drot1 = rollPointing;

        rot_3 = ra0;          # 3-rot corresponds to FOV at nominal RA.
        rot_2 = -dec0;        # 2-rot corresponds to FOV at nominal Dec. Note minus sign
                              #    since +dec corresponds to -rot

        DCM11,  DCM12,  DCM13,  DCM21,  DCM22,  DCM23,  DCM31,  DCM32, DCM33, DCM11c, DCM12c, DCM13c, DCM21c, DCM22c, DCM23c, DCM31c, DCM32c, DCM33c, chip_offset = self.get_direction_matrix(julianTime, quarter, drot1, rot_2, rot_3);

        # do conversion from pixels to RA and Dec
        quad = np.mod(output,4); # need to find quadrant in module
        # now convert from row and column to chip lng and lat
        chip_n = np.floor( (output+1)/2 +0.1).astype(int) - 1;  # chip number index  NO -1 FOR MATLAB
        
        plate_scale = self.geomModel.get("plateScale");
        plateScaleCcd = plate_scale[0:plate_scale.size:2];
        plateScaleN = plateScaleCcd[chip_n];
        plateScaleN = np.reshape(plateScaleN, chip_n.shape);
        
    #   build a vector of pincushion corrections using the same approach as the vector of
    #   plate scales
    
        pincushionAll = self.geomModel.get("pincushion");
        pincushionAlternateOutputs = pincushionAll[0:pincushionAll.size:2] ;
        pincushionN = pincushionAlternateOutputs[chip_n] ;
        pincushionN = np.reshape(pincushionN, chip_n.shape) ;
   
    
    
    #   convert the row and column coordinates to coordinates centered on the notional center
    #   of the module, with unit vectors pointing towards the readout row/column of the
    #   odd-numbered output
    
        rowCentered = rm.get_parameters("nRowsImaging") + chip_offset[chip_n,1] - row ;
        colCentered = rm.get_parameters("nColsImaging") - column ;

        evenQuadIndex = np.where((quad==0)|(quad==2));
        colCentered[evenQuadIndex] = -1 * colCentered[evenQuadIndex] ;
        colCentered = colCentered + chip_offset[chip_n,2] ;
        
    #   iteratively solve for the longitude and latitude coordinates in arcseconds, with
    #   origin at the notional center of the module; set the convergence tolerance to be
    #   somewhat larger than the eps for the class of the row variable
    
        convergenceTolerance = 8 * np.finfo(float).eps ;

        lngr, latm = self.compute_lat_long_iteratively( rowCentered, colCentered, plateScaleN, pincushionN, convergenceTolerance ) ;
        lngr = ru.make_col(np.array(lngr))
        latm = ru.make_col(np.array(latm))
    
        latm = latm/3600./rtd;  # latm in radians
        lngm = lngr/np.cos(latm);
        # correct for cos effect going from spherical to rectangular coor
        # one deg of long at one deg of lat is smaller than one deg at zero lat
        # by cos(lat) , amounts to 1/2 arc sec = 1/8 pix
        lngm = lngm/3600/rtd; # lngm in radians

        lpm = np.cos(lngm)*np.cos(latm);
        mpm = np.sin(lngm)*np.cos(latm);
        npm = np.sin(latm);

        #* transform from chip coor to FPA coor Do inverse of above transform (swap matrix row/coln indices)
        lpl = DCM11c[chip_n].reshape(chip_n.shape)*lpm + DCM21c[chip_n].reshape(chip_n.shape)*mpm + DCM31c[chip_n].reshape(chip_n.shape)*npm;
        mpl = DCM12c[chip_n].reshape(chip_n.shape)*lpm + DCM22c[chip_n].reshape(chip_n.shape)*mpm + DCM32c[chip_n].reshape(chip_n.shape)*npm;
        npl = DCM13c[chip_n].reshape(chip_n.shape)*lpm + DCM23c[chip_n].reshape(chip_n.shape)*mpm + DCM33c[chip_n].reshape(chip_n.shape)*npm;

        #* Transform from FPA to RA and Dec Again use inverse of transform matrix
        cosa = DCM11*lpl + DCM21*mpl + DCM31*npl;
        cosb = DCM12*lpl + DCM22*mpl + DCM32*npl;
        cosg = DCM13*lpl + DCM23*mpl + DCM33*npl;

        # Convert direction cosines to equatorial coordinate system
        ra  = np.arctan2(cosb,cosa)*rtd; # transformed RA in deg
        dec = np.arcsin(cosg)*rtd; # transformed Dec in deg
        ra[np.where(ra<0)] = ra[np.where(ra<0)] + 360;

        # Return column vectors
        return ra, dec
            

    def calcRaDecFromPix(self, module, output, row, column, quarter, julianTime, raPointing, decPointing, rollPointing):
        # This code only works for a constant quarter, raPointing, decPointing, and rollPointing.
        
        # check for large input array, call transformation routine in subsections if so
        #
        chunkSize = 10000;
        module = np.array(module);
        output = np.array(output);
        row = np.array(row);
        column = np.array(column);
        quarter = np.array(quarter);
        julianTime = np.array(julianTime);
        raPointing = np.array(raPointing);
        decPointing = np.array(decPointing);
        rollPointing = np.array(rollPointing);

        if module.size < chunkSize:
            ra,dec = self.doTransform_pix(module, output, row, column, quarter, julianTime, raPointing, decPointing, rollPointing)
        else:
            ra = np.zeros((module.size, 1));
            dec = np.zeros((module.size, 1));
            
            computing = True;
            dStart = 0;
            while computing:
                dEnd = dStart + chunkSize-1;
                if dEnd > module.size:
                    dEnd = module.size-1;
                    computing = False;
                dRange = np.arange(dStart,dEnd+1, dtype=int)
                dRange = np.reshape(dRange, (dEnd-dStart+1, 1))
                rTmp, dTmp = self.doTransform_pix(module[dRange], output[dRange], row[dRange], column[dRange], quarter[dRange], julianTime[dRange], raPointing[dRange], decPointing[dRange], rollPointing[dRange]);
                ra[dRange, 0] = rTmp;
                dec[dRange, 0] = dTmp;
                
                dStart = dStart + chunkSize;
                if dStart >= module.size:
                    computing = False;

        return ra, dec





    def Pix2RaDec(self, module, output, row, column, julianTime, raPointing, decPointing, rollPointing, aberrateFlag = True):
        
    #
    # function [ra, dec]  =  Pix2RaDec(raDec2PixObject, module, output, row, column, julianTime, [aberrateFlag])
    # or
    # function [ra, dec]  =  Pix2RaDec(raDec2PixObject, module, output, row, column, julianTime, raPointing, decPointing, rollPointing, [aberrateFlag])
    #
    #
    # Inputs:
    #    module - the CCD module location of the star(s).  If there is more than one
    #                 star, this argument should be a vector.  The module, output, row, and column arguments
    #                 must be the same size.
    #    output - the CCD output of the star(s).  If there is more than one
    #                 star, this argument should be a vector. The module, output, row, and column arguments
    #                 must be the same size.
    #    row   - the CCD row of the star(s).  If there is more than one
    #                 star, this argument should be a vector. The module, output, row, and column arguments
    #                 must be the same size.
    #    column - the CCD output of the star(s).  If there is more than one
    #                 star, this argument should be a vector. The module, output, row, and column arguments
    #                 must be the same size.
    #    julianTime - the julian times to do the the (RA,Dec)->Pix conversion
    #                 for.  The size of this argument must be:
    #                     1) the same as ra and dec,
    #                     2) a single value, or
    #                     3) an arbitrary length vector, iff ra and dec are
    #                        single values
    #    raPointing    - Optional argument specifying focal plane center in RA, in
    #            degrees.  raPointing, decPointing, and rollPointing must all be specified, or none.
    #    decPointing    - Optional argument specifying focal plane center in declination, in degrees.
    #    rollPointing  - Optional argument specifying focal plane rotation, in degrees.
    #    aberrateFlag - Optional argument to turn aberration on or off. The default is on.
    #            If raPointing, decPointing, and rollPointing are not specified, this can be used as the
    #            fourth argument.
    #
    #
    #
    # raPointing, decPointing, and rollPointing may be specified as 1x1 values even if other inputs are not.
    
        module = np.array(module);
        output = np.array(output);
        row = np.array(row);
        column = np.array(column);
        julianTime = np.array(julianTime);
        raPointing = np.array(raPointing);
        decPointing = np.array(decPointing);
        rollPointing = np.array(rollPointing);

        if np.any(np.isnan(module)) | np.any(np.isnan(output)) | np.any(np.isnan(row)) | np.any(np.isnan(column)) | np.any(np.isnan(julianTime)):
            raise ValueError("RaDec2Pix: nan in module, output, row, column or julianTime");
            
        if (module.size != output.size) | (module.size != row.size) | (module.size != column.size):
            raise ValueError("RaDec2Pix: module, output, row, column are not the same size");

        # Make (module, output, row, column) and julianTime the same size if one is different from
        # the other:
        #
        
        if (julianTime.size == 1) & (module.size > 1):
            julianTime = np.tile(julianTime, module.shape);

        if (module.size == 1) & (julianTime.size > 1):
            module  = np.tile(module, julianTime.shape);
            output = np.tile(output, julianTime.shape);
            row = np.tile(row, julianTime.shape);
            column = np.tile(column, julianTime.shape);
            
        quarter = self.rollTimeModel.juliandate2quarter(julianTime);
            
        # If raPointing, decPointing, and rollPointing came in as 1x1 values, stretch them to the size of
        # the other input arguments for the unique rows operation later on.
        if (raPointing.size == 1) & (module.size > 1):
            raPointing = np.tile(raPointing, module.shape);
            decPointing = np.tile(decPointing, module.shape);
            rollPointing = np.tile(rollPointing, module.shape);

        # Check to see if season, raPointing, decPointing, and rollPointing are constant
        #   if they are, then calcRaDecFromPix can be called all at once.
        #
        pointingIsConstant = (np.unique(raPointing).size == 1) & \
                             (np.unique(decPointing).size == 1) & \
                             (np.unique(rollPointing).size == 1) & \
                             (np.unique(quarter).size == 1);

        # Perform the actual geometric transformation:
        #
        if (np.unique(julianTime).size == 1) & pointingIsConstant:
            ra, dec = self.calcRaDecFromPix(module, output, row, column, quarter, julianTime, raPointing, decPointing, rollPointing);
        else:
            ra = np.zeros((julianTime.size, 1));
            dec = np.zeros((julianTime.size, 1));

            for it in range(julianTime.size):
                ra[it], dec[it] = self.calcRaDecFromPix(module[it], output[it], row[it], column[it], quarter[it], julianTime[it], raPointing[it], decPointing[it], rollPointing[it]);
            
        if aberrateFlag:
            abDat = self.aberrate.unaberrate_ra_dec(ra, dec, julianTime);
            ra = abDat[0];
            dec = abDat[1];
#        print("ra = " + str(ra))
#        print("dec = " + str(dec))
        return [ra, dec]




    def pix_2_ra_dec(self, module, output, row, column, mjds,
                raPointing = None, decPointing = None, rollPointing = None,
                deltaRa = None, deltaDec = None, deltaRoll = None, aberrateFlag = True):
        

        module = ru.make_col(np.array([module]))
        output = ru.make_col(np.array([output]))
        row = ru.make_col(np.array([row]))
        column = ru.make_col(np.array([column]))
        mjds = ru.make_col(np.array([mjds]))
             
        isReturnMatrix = (module.size > 1) & (mjds.size > 1);

        jds = mjds + rm.get_parameters("mjdOffset");

        ra = np.zeros((module.size, mjds.size));
        dec = np.zeros((module.size, mjds.size));
        
        if (np.all(raPointing == None)) | (np.all(raPointing == -1)):
            pointing = self.pointingModel.get_pointing(mjds);
            raPointing = pointing[0];
            decPointing = pointing[1];
            rollPointing = pointing[2];
        else:
            raPointing = ru.make_col(np.array([raPointing]))
            decPointing = ru.make_col(np.array([decPointing]))
            rollPointing = ru.make_col(np.array([rollPointing]))
            
        if np.any(deltaRa != None):
            raPointing = raPointing + ru.make_col(np.array([deltaRa]));
            decPointing = decPointing + ru.make_col(np.array([deltaDec]));
            rollPointing = rollPointing + ru.make_col(np.array([deltaRoll]));

        
        # Adjust inputs to move center of pixel to (0.0, 0.0);
        #
        row = row + 0.5;
        column = column + 0.5;

        # Adjust the pixel inputs in CCD coordinates to the coordinates on visible silicon:
        #
        row    = row    - rm.get_parameters("nMaskedSmear");
        column = column - rm.get_parameters("nLeadingBlack");

        if isReturnMatrix:
            for itime in range(jds.size):
                pixData = self.Pix2RaDec(module, output, row, column, jds[itime], raPointing[itime], decPointing[itime], rollPointing[itime], aberrateFlag);
                ra[:,itime] = pixData[0].ravel();
                dec[:,itime] = pixData[1].ravel();
        else:
            pixData = self.Pix2RaDec(module, output, row, column, jds, raPointing, decPointing, rollPointing, aberrateFlag);
            ra = pixData[0];
            dec = pixData[1];

        if ra.size == 1:
            return ra[0,0], dec[0,0]
        elif ra.shape[1] == 1:
            return ra[:,0], dec[:,0]
        else:
            return ra, dec

        
