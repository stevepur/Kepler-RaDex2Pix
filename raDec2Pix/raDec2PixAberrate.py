
import spiceypy as spice
from astropy.time import Time
import numpy as np
import array
from raDec2Pix import raDec2PixUtils as ru
import os

_dataPath = os.path.join(os.path.dirname(__file__), "data")

class aberrateRaDec:        
    def get_state(self, julianTime):
        # utc is an array of julian dates
        spice.furnsh(_dataPath + "/naif0012.tls")
        spice.furnsh(_dataPath + "/spk_2018127000000_2018128182705_kplr.bsp")
        spice.furnsh(_dataPath + "/de421.bsp")

        if np.isscalar(julianTime):
            julianTime = ru.make_col(np.array([julianTime]))
        if julianTime.ndim == 2:
            julianTime = julianTime.reshape((julianTime.size,))
        julianTimes = Time(julianTime, format='jd', scale='utc')
        times = spice.str2et(julianTimes.iso)
        
        # in case the input is just a float with no len
        if not hasattr(times, "__len__"):
            times = [times]
            
        state = np.zeros((len(times),6))
        for i in range(len(times)):
            state[i,:], lightTimes = spice.spkezr('Kepler', times[i], 'J2000', 'NONE', 'SUN')
        
        spice.kclear()
        return state

    def get_velocity_vector_mps(self, julianTime):
        state = self.get_state(julianTime)
        if hasattr(julianTime, "__len__"):
            return 1000.*state[:,3:6].reshape((len(julianTime),3))
        else:
            return 1000.*state[0,3:6]
            
    def get_position_vector_m(self, julianTime):
        state = self.get_state(julianTime)
        if hasattr(julianTime, "__len__"):
            return 1000.*state[:,0:3].reshape((len(julianTime),3))
        else:
            return 1000.*state[0,0:3]
            
    def convert_stars_sph2cart(self, ra, dec):
        d2r = np.pi/180;
        ra = np.array(ra);
        dec = np.array(dec);

        az = d2r * ra;
        elev = d2r * dec;
        r = np.ones((ra.size, 1))

        z = r * np.sin(elev);
        rcoselev = r * np.cos(elev);
        x  = rcoselev * np.cos(az);
        y  = rcoselev * np.sin(az);

        return x, y, z


    def convert_stars_cart2sph(self, x, y , z):

        x = np.array(x);
        y = np.array(y);
        z = np.array(z);

        raDec = np.zeros((x.size,2));

        hypotxy = np.hypot(x,y);
        r = np.hypot(hypotxy,z); # r
        elev = np.arctan2(z,hypotxy); # elev
        az = np.arctan2(y,x); # az
        
        r2d = 180/np.pi;

        ra = ru.bound(r2d*az, 0, 360) # ra
        dec = ru.bound(r2d*elev, -90, 90) # dec
        
        ra = ru.make_col(ra);
        dec = ru.make_col(dec);
        return [ra, dec]
        
    def vel_aber(self, x, y, z, velocity):
    #    %
    #    % function [xnew, ynew, znew] = vel_aber(x, y, z, velocity)
    #    %
    #    % Inputs:
    #    %     x, y, z: Row vectors of the cartesian vectors towards nStars stars, where
    #    %              nStars is the number of columns in the vectors.  These x,y,z
    #    %              triplets need not be normalized.
    #    %
    #    %     velocity:  A matrix of nTimes rows by 3 columns.  Each row represents
    #    %                the spacecraft velocity in in meters per second at that
    #    %                point in time.
    #    %
    #    % Output:
    #    %     Three matricies representing the cartesian direction towards nStars
    #    %     stars.  Each matrix will be nTimes rows x nStars columns, and the
    #    %     triplet x(iTime, iStar), y(iTime, iStar), z(iTime, iStar) is the
    #    %     apperent position of star iStar at time iTime.
        x = np.array(x);
        y = np.array(y);
        z = np.array(z);
        velocity = np.array(velocity);

        nStars = x.size;
        if velocity.ndim == 1:
            nTimes = 1;
        else:
            nTimes = velocity.shape[0];
        pt = np.column_stack((ru.make_col(x), ru.make_col(y), ru.make_col(z)));
        pt = pt/ru.make_col(np.linalg.norm(pt, axis=1)); # make rows unit vectors

        lightspeed    = 2.99792458e8;  # meters/second
        vel           = velocity / lightspeed;

        # Black magic from Jon to to reshape bigVel into
        # the same format as bigPt
        #
        bigPt  = np.tile(pt, (nTimes, 1)); # nStars*nTimes x 3, a copy of nStars x 3 for each time
        bigVel = np.tile(vel,(nStars,1,1)); # nStars x nTimes x 3, make a copy of velocity for each star
        bigVel = np.reshape(bigVel,(nTimes*nStars,3)); # nStars*nTimes x 3, group together, so each time's velocity appears twice, once for each star

        oneOverBeta = ru.make_col(np.sqrt(1 - np.sum(bigVel*bigVel, 1)));
        pDotV       = ru.make_col(np.sum(bigPt*bigVel, 1));

        k1 = np.tile(oneOverBeta,                    (1, 3));
        k2 = np.tile(1 + pDotV / (1 + oneOverBeta), (1, 3));
        k3 = np.tile(1. / (1 + pDotV),               (1, 3));

        apparentPt = k3 * (k1 * bigPt + k2 * bigVel);
        if x.size > 1:
            apparentPt = apparentPt/ru.make_col(np.linalg.norm(apparentPt, axis=1)); # make rows unit vectors

        xnew = np.transpose(np.reshape(apparentPt[:,0], (nStars, nTimes)));
        ynew = np.transpose(np.reshape(apparentPt[:,1], (nStars, nTimes)));
        znew = np.transpose(np.reshape(apparentPt[:,2], (nStars, nTimes)));
        
        return xnew, ynew, znew
        
    def aberrate_ra_dec(self, ra, dec, julianTime):
        # function [aberRa, aberDec] = aberrate_ra_dec(ra, dec, julianTime, [vel])
        #   Apply an aberration to the RA and Dec based on the spacecraft state
        #   vector at time julianTime.  If there's only one julianTime, it is used for
        #   all RAs and Decs, otherwise, ra, dec, and julianTime should all have the
        #   same number of elements, and the aberration will be applied one-to-one.
        vel = self.get_velocity_vector_mps(julianTime);
        
        x, y, z = self.convert_stars_sph2cart(ra, dec);

        xab, yab, zab = self.vel_aber( x, y, z, vel );
        return self.convert_stars_cart2sph(xab, yab, zab);

    def vel_aber_inv(self, x, y, z, velocity):
        #
        #function [xnew, ynew, znew] = vel_aber_inv(x, y, z, velocity, bRelativistic)
        #
        # Inputs:
        #     x,y,z: Three vectors representing the actual direction to the star.  Need
        #         not be normalized.
        #
        #     velocity:  a column of one or more three-vectors representing the
        #         velocity of the observer, in meters per second.  Must have
        #         the same dimensions as pt.
        #
        #     bRelativistic: A boolean flag to determine if the calculation is done
        #         using the Newtonian or the relativistic form.  Defaults to 1.  The
        #         Newtonian form may be faster.
        #
        # Output:
        #     Three vectors with the same dimension as the inputs. The 3 vectors
        #         are components of the unit vectors representing the true direction of
        #         the star.
        #
        apparent_pt = np.array([x[0], y[0], z[0]]);

        lightspeed    = 2.99792458e8;
        vel           = velocity / lightspeed;
        if vel.ndim == 1:
            beta = np.linalg.norm(vel);
        else:
            beta = np.linalg.norm(vel, axis=1);

        # angle between velocity vector and apparent direction -- NB that this calculation is of
        # limited and possibly unacceptable accuracy for alpha prime within ~1 degree of zero.
        alpha_prime = np.arccos(np.sum(apparent_pt*ru.unitv(velocity)));

        # initialize solution: set actual angle to apparent angle:
        alpha = alpha_prime;

        alpha_old = alpha+1; # initialize to get into loop

        iterCount = 0;
        maxIter = 1000;
        while np.any(np.abs(alpha-alpha_old)>1e-12):
            alpha_old = alpha;
            dalpha = (np.tan(alpha_prime)*np.cos(alpha)+beta*np.tan(alpha_prime)-np.sqrt(1-beta**2)*np.sin(alpha)) \
                /(np.tan(alpha_prime)*np.sin(alpha)+np.sqrt(1-beta**2)*np.cos(alpha));
            alpha=alpha+dalpha;
            iterCount = iterCount + 1;
            if iterCount > maxIter:
                raise ValueError("alpha did not converge");


        # find normal to velocity vector
        unit_vel=ru.unitv(vel);
        normal = np.cross(unit_vel,apparent_pt); # out of paper vector
        normal = ru.unitv(np.cross(normal,unit_vel));

        pt = np.tile(np.cos(alpha),(1,3))*unit_vel+np.tile(np.sin(alpha),(1,3))*normal;
        pt = ru.unitv( pt );

        xnew = pt[0,0];
        ynew = pt[0,1];
        znew = pt[0,2];
        return xnew, ynew, znew

    def unaberrate_ra_dec(self, ra, dec, julianTime):
        # function [aberRa, aberDec] = unaberrate_ra_dec(ra, dec, julianTime)
        #   Remove an aberration to the RA and Dec based on the spacecraft state
        #   vector at time julianTime.  If there's only one julianTime, it is used for
        #   all RAs and Decs, otherwise, ra, dec, and julianTime should all have the
        #   same number of elements, and the aberration will be applied one-to-one.

        raApparent = ra
        decApparent = dec
        
        nRaDec = raApparent.size;
        
        velocity = self.get_velocity_vector_mps(julianTime);
        if velocity.shape[0] == 1:
            velocity = np.tile(velocity, raApparent.shape);
        
        # convert the ra and dec into a Cartesian vector which points at the point of interest in
        # an equatorial coordinate system
        x, y, z = self.convert_stars_sph2cart(raApparent, decApparent);
        
        # construct vectors of unabberated Cartesian coordinates which will be filled in by the
        # calculation engine

        xActual = np.zeros(x.shape) ;
        yActual = np.zeros(x.shape) ;
        zActual = np.zeros(x.shape) ;
          
        # perform the calculation:  since vel_aber_inv is not vectorized, the calculation must be
        # made for one point and one velocity at a time, via the dreaded for-loop
        for iRaDec in range(nRaDec):
            xActual[iRaDec], yActual[iRaDec], zActual[iRaDec] = self.vel_aber_inv( x[iRaDec], y[iRaDec], z[iRaDec], velocity[iRaDec,:] ) ;
          
        # convert back to RA and Dec from Cartesian coordinates
        return self.convert_stars_cart2sph(xActual, yActual, zActual);

