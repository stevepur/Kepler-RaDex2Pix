# Kepler-RaDex2Pix
 Port of Kepler raDec2Pix code from Matlab to Python

raDec2Pix (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6349250/) was a Matlab class used to map sky position to pixel postion for the Kepler space telescope.  It provides high-acccuracy, accounting for nominal Kepler pointing, differential velocity aberration and Kepler as-built CCD geometry.  It does not account for thermally-induced focus variations (leading to changes in plate scale) or the residual roll motion during K2.  Errors due to the focus variations can be as large as a half pixel at the edge of the focal plane in some Kepler quarters.  Errors due to residual roll motion in K2 can be 2 or more pixels at the edge of the focal plane.

This python port duplicates the Matlab code's behavior, and supports both the original Kepler mission and K2.  

The Python notebooks raDec2Pix_example.ipynb and test_ra_dec_2_pix.ipynb provide usage examples.

This class contains the following functions:

Initializer:

.raDec2PixClass(dataDir): dataDir is the relative path to the raDec2PixDir directory, containing the data files required by raDec2Pix.

Function that maps RA and DEC to pixel position at a given time:

m, o, r, c = .ra_dec_2_pix(ra, dec, mjds, raPointing = None, decPointing = None, rollPointing = None, deltaRa = None, deltaDec = None, deltaRoll = None, aberrateFlag = True).  

Inputs: 
- ra, dec: RA and DEC position in degrees.  May be single values or arrays.  If arrays they must have the same length.  The valid ranges are 0 <= ra <= 360, -90 < dec < 90.
- mjds: time in modified Julian day.  May be single value or an array.  The valid mjd range is from 54943.152 (2009-04-22 03:38:52.800) to 58482.652 (2018-12-30 15:38:52.800), which includes K2.
If ra, dec, and mjds are all arrays, they must be the same size.
- raPointing, decPointing, rollPointing: Kepler pointing in degrees to be used in the conversion.  If they are None (default), the nominal (commanded) mission pointing is used, computed for the date(s) given in mjds. The actual pointing may be slightly different from the commanded pointing due to tracking errors. If supplied, these must have the same size as mjds. 
- deltaRa, deltaDec, deltaRoll: pointing offsets in degrees that is added to the (nominal or supplied) pointing to determine the Kepler pointing.  The most common use case is to supply these deltas and not the pointing, so the deltas are added to the nominal pointing. If supplied, they must have the same size as mjds.
- aberrateFlag: boolean that determines whether RA and DEC should be aberrated according to Kepler's orbital motion.  For actual flight conditions this should be set to True.

Outputs:
- m: CCD module that the principal optical ray falls on. Negative values indicate that the sky position does not fall on any CCD.
- o: CCD output that the principal optical ray falls on. Negative values indicate that the sky position does not fall on any CCD.
- r: row where the principal optical ray hits the CCD. Note that only rows 20 through 1043 fall on the visible portion of the CCD.
- c: column where the principal optical ray hits the CCD. Note that only colums 12 through 1111 fall on the visible portion of the CCD.

If ra, dec and mjds are single values, then m, o, r and c are single values.

If ra and dec are single values and mjds is an array, then m, o, r and c are arrays with the same size as mjds, containing the pixel positions of the single ra,dec pair at each mjd.

If ra and dec are arrays and mjds is a single value, then m, o, r and c are arrays with the same size as ra and dec, containing the pixel positions for each ra,dec pair at each mjd.

if ra, dec and mjds are arrays, then every element of mjds is applied to each ra,dec pair, and m, o, r and c are matrices.  The diagonal of the returned matrices are the ra and dec at their corresponding mjds.




