# Kepler-RaDex2Pix
 Port of Kepler raDec2Pix code from Matlab to Python

raDec2Pix (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6349250/) was a Matlab class used to map sky position to pixel postion (and vice versa) for the Kepler space telescope.  It provides high-acccuracy, accounting for nominal Kepler pointing, differential velocity aberration and Kepler as-built CCD geometry.  It does not account for thermally-induced focus variations (leading to changes in plate scale) or the residual roll motion during K2.  Errors due to the focus variations can be as large as a half pixel at the edge of the focal plane in some Kepler quarters.  Errors due to residual roll motion in K2 can be 2 or more pixels at the edge of the focal plane.

This python port duplicates the Matlab code's behavior, and supports both the original Kepler mission and K2.  

The Python notebooks raDec2Pix_example.ipynb, test_ra_dec_2_pix.ipynb and test_pix_2_ra_dec.ipynb provide usage examples.

ra_dec_2_pix is fast when called with arrays of (ra, dec) pairs for a single mjd, processing 12.5 million positions in 4.5 minutes.  pix_2_ra_dec is slower at a single mjd, processing one million positions in 9 minutes.

## This class contains the following functions:

### Initializer:

.raDec2PixClass(dataDir): dataDir is the relative path to the raDec2PixDir directory, containing the data files required by raDec2Pix.

### Function that maps RA and DEC to pixel position at a given time:

m, o, r, c = .ra_dec_2_pix(ra, dec, mjds, raPointing = None, decPointing = None, rollPointing = None, deltaRa = None, deltaDec = None, deltaRoll = None, aberrateFlag = True).  

#### Inputs: 
- ra, dec: RA and DEC position in degrees.  May be single values or arrays.  If arrays they must have the same length.  The valid ranges are 0 <= ra <= 360, -90 < dec < 90.
- mjds: time in modified Julian day.  May be single value or an array.  The valid mjd range is from 54943.152 (2009-04-22 03:38:52.800) to 58482.652 (2018-12-30 15:38:52.800), which includes K2.
If ra, dec, and mjds are all arrays, they must be the same size.
- raPointing, decPointing, rollPointing: Kepler pointing in degrees to be used in the conversion.  If they are None (default), the nominal (commanded) mission pointing is used, computed for the date(s) given in mjds. The actual pointing may be slightly different from the commanded pointing due to tracking errors. If supplied, these must have the same size as mjds. 
- deltaRa, deltaDec, deltaRoll: pointing offsets in degrees that is added to the (nominal or supplied) pointing to determine the Kepler pointing.  The most common use case is to supply these deltas and not the pointing, so the deltas are added to the nominal pointing. If supplied, they must have the same size as mjds.
- aberrateFlag: boolean that determines whether RA and DEC should be aberrated according to Kepler's orbital motion.  For actual flight conditions this should be set to True.

#### Outputs:
- m: CCD module that the principal optical ray falls on. Negative values indicate that the sky position does not fall on any CCD.
- o: CCD output that the principal optical ray falls on. Negative values indicate that the sky position does not fall on any CCD.
- r: row where the principal optical ray hits the CCD. Note that only rows 20 through 1043 fall on the visible portion of the CCD.
- c: column where the principal optical ray hits the CCD. Note that only colums 12 through 1111 fall on the visible portion of the CCD.

If ra, dec and mjds are single values, then m, o, r and c are single values.

If ra and dec are single values and mjds is an array, then m, o, r and c are arrays with the same size as mjds, containing the pixel positions of the single ra,dec pair at each mjd.

If ra and dec are arrays and mjds is a single value, then m, o, r and c are arrays with the same size as ra and dec, containing the pixel positions for each ra,dec pair at the specified mjd.

if ra, dec and mjds are arrays, then every element of mjds is applied to each ra,dec pair, and m, o, r and c are matrices.  The diagonal of the returned matrices are the m, o, r and c at their corresponding mjds.

Example:
<pre><code>
import raDec2Pix
rdp = raDec2Pix.raDec2PixClass("raDec2PixDir")
m, o, r, c = rdp.ra_dec_2_pix(299.89509, 40.6334, 55183.0)
print("module " + str(m) + ", output " + str(o) + ", row " + str(r) + ", column " + str(c))

module 22, output 3, row 162.8879188939173, column 99.7341745175919
</code></pre>

### Function that maps pixel position to RA and DEC at a given time:

ra, dec = .pix_2_ra_dec(mod, out, row, column, mjds, raPointing = None, decPointing = None, rollPointing = None, deltaRa = None, deltaDec = None, deltaRoll = None, aberrateFlag = True).  

#### Inputs: 
- mod: Module containing the pixel on the Kepler CCD array.  May be single values or arrays.  If arrays they must have the same length as out, row and column.  Valid values are in the set [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24] ([2:4, 6:20, 22:24]). 
- out: Output containing the pixel on the module specified in mod.  May be single values or arrays.  If arrays they must have the same length as mod, row and column.  Valid values are 1, 2, 3 or 4.
- row, column: the pixel row and column on the module and output specified by mod and out.  May be single values or arrays.  If arrays they must both have the same length, and the same length as as out and mod.  Valid values are 0 <= row <= 1069 and 0 <= column <= 1131.
- mjds: time in modified Julian day.  May be single value or an array.  The valid mjd range is from 54943.152 (2009-04-22 03:38:52.800) to 58482.652 (2018-12-30 15:38:52.800), which includes K2.
If ra, dec, and mjds are all arrays, they must be the same size.
- raPointing, decPointing, rollPointing: Kepler pointing in degrees to be used in the conversion.  If they are None (default), the nominal (commanded) mission pointing is used, computed for the date(s) given in mjds. The actual pointing may be slightly different from the commanded pointing due to tracking errors. If supplied, these must have the same size as mjds. 
- deltaRa, deltaDec, deltaRoll: pointing offsets in degrees that is added to the (nominal or supplied) pointing to determine the Kepler pointing.  The most common use case is to supply these deltas and not the pointing, so the deltas are added to the nominal pointing. If supplied, they must have the same size as mjds.
- aberrateFlag: boolean that determines whether RA and DEC should be aberrated according to Kepler's orbital motion.  For actual flight conditions this should be set to True.

Invalid input values of mod, out, row or column return invalid results but are not trapped on input (following the behavior of the original function), and may cause the code to crash.  

#### Outputs:
- ra, dec: the right ascension and declination of the sky postion whose principle ray falls at mod, out, row, column on the date(s) specified by mjds.

If mod, out, row, column and mjds are single values, then ra and dec are single values.

If mod, out, row, column are single values and mjds is an array, then ra and dec are arrays with the same size as mjds, containing the pixel positions of the single ra,dec pair at each mjd.

If mod, out, row, column are arrays and mjds is a single value, then ra and dec are arrays with the same size as mod, out, row, column, containing the pixel positions for each mod, out, row, column set at the single mjd.

if mod, out, row, column and mjds are arrays, then every element of mjds is applied to each mod, out, row, column set, and ra, dec are matrices.  The diagonal of the returned matrices are the ra and dec at their corresponding mjds.

Example:
<pre><code>
import raDec2Pix
rdp = raDec2Pix.raDec2PixClass("raDec2PixDir")
ra,dec = rdp.pix_2_ra_dec(22, 3, 162.9, 99.7, 55183.0)
print("ra " + str(ra) + ", dec " + str(dec))

ra 299.8951264356314, dec 40.63342892949698
</code></pre>

