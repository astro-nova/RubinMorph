import sys
import os
import logging
import galsim
import copy
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps
from astropy.visualization import simple_norm
import petrofit
from astropy.modeling.functional_models import Gaussian2D
import matplotlib.pyplot as plt
from importlib import resources as impresources
import templates
from skimage.transform import swirl

gsparams = galsim.GSParams(maximum_fft_size=20000)

#################### RNG set-up ###########################
# TODO: clump brightness should be in terms of surface brightness not total flux
# to give more flux to bigger ones so they don't get lost

_default_clump_properties = {
    'r' : (0.05, 1.0),
    'logf' : (-3, -1), # From 1% to 10% of the tital light
    'logs' : (-1.5, 0.5) 
}


_default_galaxy_properties = {
    'mag' : (11, 18),
    'n_clumps' : (10, 100),
    'sersic_n' : (1, 5),
    'q' : (0.1, 1),
    'reff_scatter' : (0, 1.5) # Mean and standard deviation
}


_default_aug_properties = {
    'sky_mag' : (20, 30),
    'pxscale' : (0.1, 2/3)
}

companion_threshold = 0.3
_default_companion_properties = {
    'r' : (0.7, 1.5),
    'n' : (1, 3),
    'q' : (0.1, 1),
    'r_frac' : (0.2, 0.5),
    'flux_frac' : (0.3, 0.7)
}

#################### SDSS set-up ###########################
sdss_ra = 150
sdss_dec = 2.3
filt = 'r'
bandpass_file = "passband_sdss_" + filt
inp_file = (impresources.files(templates) / bandpass_file)
throughput = galsim.LookupTable.from_file(inp_file)
bandpass = galsim.Bandpass(throughput, wave_type = u.angstrom)
## gain, exptime and diameter of telescope
# telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
telescope_params = {'g':4.8, 't_exp': 53.91*1000, 'D':2.5}
## effective wavelength and width of filter
transmission_params = {'eff_wav':616.5, 'del_wav':137}
g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
transmission = bandpass(transmission_params['eff_wav'])
######### ZEROPOINT: 38

#############################################################


def mag2uJy(mag):
    """
    helper function to go from mag to uJy

    Args:
        mag (float) : AB magnitude value
    Returns:
        flux density in uJy 
    """
    return 10**(-1*(mag-23.9)/2.5)

def mag2nmgy(mag):
    """
    helper function to go from mag to nmgy

    Args:
        mag (float) : AB magnitude value
    Returns:
        flux density in nanomaggies
    """
    return 10**(-1*(mag-22.5)/2.5)

def uJy2galflux(uJy, lam_eff, lam_del, throughput):
    """
    helper function to go from uJy to flux in electrons/cm^2/s using total throughput

    Args:
        uJy (float) : flux density in uJy
        lam_eff (float) : effective wavelength in nanometers
        lam_del (float) : FWHM of throughput curve in nanometers
        throughput (float) : transmission value at lam_eff
    Returns:
        flux value in electrons/s/cm^2

    """
    lam_eff *= 1e-9  # convert from nm to m
    lam_del *= 1e-9  # convert from nm to m
    nu_del = 2.998e8 / (lam_eff ** 2) * lam_del  # difference in wavelength to difference in frequency
    lum_den = (uJy * u.uJy).to(u.photon / u.cm**2 / u.s / u.Hz, equivalencies=u.spectral_density(lam_eff * u.m)).value
    return throughput * lum_den * nu_del 

def gen_image(centre_ra, centre_dec, pixel_scale, fov_x, fov_y):
    """
    Generate image with wcs info
    
    Args:
        centre_ra (float) : right ascension in deg
        centre_dec (float)  : declination in deg
        pixel_scale (float) : in arcsec/pixel
        fov_x (float) : in deg
        fov_y (float) : in deg
    Returns:
        image (galsim object) : galsim image object with wcs info and fov
        wcs (wcs object) : WCS header for image 
    """
    centre_ra_hours = centre_ra/15.
    cen_ra = centre_ra_hours * galsim.hours
    cen_dec = centre_dec * galsim.degrees

    cen_coord = galsim.CelestialCoord(cen_ra, cen_dec)

    image_size_x = round(fov_x*3600/pixel_scale)
    image_size_y = round(fov_y*3600/pixel_scale)
    image = galsim.Image(image_size_x, image_size_y)

    affine_wcs = galsim.PixelScale(pixel_scale).affine().shiftOrigin(image.center)
    wcs = galsim.TanWCS(affine_wcs, world_origin = cen_coord)
    image.wcs = wcs
    ix = int(image.center.x)
    iy = int(image.center.y)
    
    return image, wcs

def gen_galaxy(mag, re, re_second, q, beta, n_disk, n_bulge, b_t):
    """
    create a sersic profile galaxy with given mag, re, n, q, beta

    Args:
        mag (float) : AB magnitude of galaxy
        re (float) : effective radius in arcsec
        re_second (float): radius of the second component in arcsec
        q (float) : axis ratio of galaxy
        beta (float) : position angle of galaxy
        n_disk (float): Sersic index of the disk component (1<n<2.5)
        n_bulge (flaot): Sersic index of the bulge component (2.5<n<5)
        b_t (float): Bulge-to-total flux fraction (0<b/t<1)
    Returns:
        gal (galsim object) : galsim galaxy object
    """
    g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
    eff_wav, del_wav = transmission_params['eff_wav'],transmission_params['del_wav']

    transmission = bandpass(transmission_params['eff_wav'])

    uJy = mag2uJy(mag)
    
    ## flux in electrons
    flux = uJy2galflux(uJy, eff_wav, del_wav, transmission) * t_exp * np.pi * (D*100./2)**2

    # Decide which component is the primary 
    if b_t > 0.5:
        r_bulge = re
        r_disk = re_second 
    else:
        r_bulge = re_second 
        r_disk = re 

    # re is circular re, can then shear the galaxy with an axis ratio and angle
    disk = galsim.Sersic(n=n_disk, flux=flux*(1-b_t), half_light_radius=r_disk, gsparams=gsparams)
    t = [[np.cos(beta), -q*np.sin(beta)], [np.sin(beta), q*np.cos(beta)]]
    disk = disk.transform(t[0][0], t[0][1], t[1][0], t[1][1])/np.abs(np.linalg.det(t))
    # disk = disk.shear(q = q, beta=-1*beta*galsim.radians)
    bulge = galsim.Sersic(n=n_bulge, flux=flux*b_t, half_light_radius=r_bulge, gsparams=gsparams)
    gal = disk+bulge

    return gal

def sky_noise(image_psf, sky_mag, pxscale, seed=None, rms_noise=True, **kwargs):
    """
    take image and sky level, calculate level in electrons and apply noise with sky level and source e counts
    can be seeded

    Args:
        image_psf (array) : 2D numpy array
        sky_mag (float) : mag/arcsec^2 value of sky background
        pixel_scale (flaot) : arcsec/pixel
        telescope_params (dict) : telescope parameters (gain, exptime and mirror diameter)
        transmission_params (dict) : tramission parameters (effective wavelength and width)
        bandpass (galsim obbject) : galsim bandpass object defining the total throughput curve
        seed (int) : seed value for noise (default None)
        rms_noise (bool): whether the sky_mag is the brightness of the sky (default, False) or its RMS brightness (True)
    Returns:
        image_noise (galsim object) : image_psf + addded noise (image_psf is preserved due to copying)
    """

    sky_uJy = mag2uJy(sky_mag)*pxscale*pxscale
    sky_electrons = uJy2galflux(sky_uJy, transmission_params['eff_wav'], 
                 transmission_params['del_wav'], transmission) * t_exp * np.pi * (D*100./2)**2

    # If noise is given as background rms instead of background brightness
    if rms_noise: sky_electrons = sky_electrons**2

    # copy image in case iterating over and changing noise level
    if 'galsim' in str(type(image_psf)):

        # print('galsim')

        rng = galsim.BaseDeviate(seed) # if want to seed noise
        image_noise_setup = image_psf.copy()
        # print(np.min(image_noise.array))
        image_noise_setup.addNoise(galsim.PoissonNoise(rng=rng, sky_level=sky_electrons))

        image_noise = image_noise_setup.array
    

    elif type(image_psf) == np.ndarray:
        # print('astropy')
        image_noise_setup = image_psf.copy()
        # # print(np.min(image_noise_setup), sky_electrons)
        image_noise_setup += sky_electrons
        # # print(np.min(image_noise_setup), '\n')
        image_noise = np.random.poisson(lam=image_noise_setup).astype(float)
        # Testing only: no Poisson noise
# 		image_noise = image_noise_setup + np.random.normal(0, np.sqrt(sky_electrons), size=image_noise_setup.shape)
# 		return image_noise, np.sqrt(sky_electrons)
        # print('here')
        image_noise -= sky_electrons

        

    else:
        print('Type not understood')

    return image_noise, np.sqrt(sky_electrons)

def petrosian_sersic(fov, re, n):
    """
    calculate r_p based on sersic profile
    Args:
        fov (float) : just a stopping point for the range of r_vals [deg]
        re (float) : effective radius in arcsec
        n (float) : sersic index
    Returns:
        PR_p2 : Petrosian radius in arcsec
    """
    fov = fov*3600
    R_vals = np.arange(0.001, fov/2.0, 0.5)
    R_p2_array = petrofit.modeling.models.petrosian_profile(R_vals, re, n)
    R_p2 = R_vals[np.argmin(np.abs(R_p2_array-0.2))]
    return R_p2

def create_clumps(image, rp, gal_mag, q, beta, clump_properties=None):
    """Create gaussian clumps to add to galaxy image to simulate intrinsic asymmetry.
    Note that now all values for clumps are generated in get_galaxy_rng_vals
    to make consistensy checks easier.

    Args:
        image (galsim.Image): galaxy image generated by galsim
        rp (float): petrosian radius in pixels
        N (int): number of clumps to generate
        gal_mag (float): magnitude of the main source
        telescope_params, transmission_params, bandpass: telescope imaging parameters for galsim
        clump_properties (dict):
            Some or all clumps can be defined by the user using clump_properties.
            This should be a dict containing keys `r`, `theta`, `flux`, and `sigma`.
                r (list): Mx1 list of positions as a function of R_pet, M<=N.
                theta (list): Mx1 list of clump angles w.r.t. x-axis
                flux (list): Mx1 list of clump fluxes as a fraction of main object flux
                sigma (list): Mx1 list of clump sizes in pixels
            M clumps will be generated with these properties. The remaining N-M will be
            generated randomly. If None, all clumps will be randomly generated.
        DEPRECATED: random_clump_properties (dict):
            Dict of tuples definining minimum and maximum values for each of the properties
            needed to generate random clumps.
                r (tuple): (rmin, rmax) as a function of petrosian radius
                flux (tuple): (fmin, fmax) as a function of main object flux
                sigma (tuple): (sigma_min, sigma_max) range of possible clump sizes.
            If None or some keys are missing, default values are used instead:
            (0, 1) for r, (0.05, 0.1) for flux, (0.5, 5) for sigma.
    """


    # getting galaxy flux from mag
    uJy = mag2uJy(gal_mag)
    flux = uJy2galflux(uJy, transmission_params['eff_wav'], 
                    transmission_params['del_wav'], transmission)/g * t_exp * np.pi * (D*100./2)**2

    # Check that random clump properties are set. If not, set the default ones instead.
    # if not random_clump_properties: random_clump_properties = _default_clump_properties
    # for key in ['r', 'flux', 'sigma']:
    # 	if key not in random_clump_properties:
    # 		random_clump_properties[key] = _default_clump_properties[key]

    # get center of image and generate possible pixel values for clumps within r_p
    # TODO: technically this can place clumps as far as Rp * sqrt(2), could convert to polar
    # to make this limited by Rp
    xc = image.center.x
    yc = image.center.y

    clumps = []
    all_xi = []
    all_yi = []
    
    # Make sure the dictionary is in the correct order to avoid bugs
    dict_keys = ['r', 'theta', 'flux_frac', 'sigma']
    # Pull lists for each property out of the dictionary
    props = [clump_properties[k] for k in dict_keys]
    # Iterate through properties of each clump. This enforces the lists have the same length.
    for r, th, flux_frac, sig in zip(*props):

        # Get position in terms of (r, theta) and convert to (x, y)
        # x = round(xc+r*rp*np.cos(th))
        # y = round(yc+r*rp*np.sin(th))
        # Embed the clumps within the inclined galaxy
        x = r*rp*np.cos(th)
        y = r*rp*np.sin(th)
        xrot = x*np.cos(beta) - y*q*np.sin(beta)
        yrot = x*np.sin(beta) + y*q*np.cos(beta)
        x = round(xc+xrot)#r*rp*np.cos(th))
        y = round(yc+yrot)#r*rp*np.sin(th))


        # create clump
        clump = galsim.Gaussian(flux=flux*flux_frac, sigma=sig, gsparams=gsparams)
        # Incline clumps with the galaxy
        t = [[np.cos(beta), -q*np.sin(beta)], [np.sin(beta), q*np.cos(beta)]]
        clump = clump.transform(t[0][0], t[0][1], t[1][0], t[1][1])/np.abs(np.linalg.det(t))
        # clump = clump.shear(q = 0.5, beta=-1*galsim.radians)

        # add to lists
        clumps.append(clump)
        all_xi.append(x)
        all_yi.append(y)

    # Create a "tail" i.e. a feature on the outskirts
    # x = round(xc+0.7*rp)
    # y = round(yc+0.5*rp)
    # clump = galsim.Sersic(flux=flux*0.6, n=1, half_light_radius=10)
    # clump = clump.shear(q = 0.6, beta=-0.5*galsim.radians)
    # clumps.append(clump)
    # all_xi.append(x)
    # all_yi.append(y)

    return clumps, all_xi, all_yi

def add_companion(image, rp, gal_mag, pxscale, companion_props=None):
    """Create a Sersic profile slightly removed from the main galaxy.
    All values for clumps are generated in get_galaxy_rng_vals.

    Args:
        image (galsim.Image): galaxy field image generated by galsim
        rp (float): petrosian radius in pixels
        gal_mag (float): magnitude of the main source
        companion_props (dict):
            This should be a dict containing keys `r`, `theta`, `flux_frac`, `r_frac`, 'q', 'n', 'beta'
                r (float): position as a function of R_pet
                theta (float): comopanion angular position w.r.t. x-axis
                flux_frac (float): flux as a fraction of main object flux
                r_frac (float): effective radius as a fraction of R_pet
                n (float): Sersic index of the companion, 1<n<3
                q (float): Axis ratio of the companion, 0<q<1
                beta (float): Inclination of the companion
    """


    # getting galaxy flux from mag
    uJy = mag2uJy(gal_mag)
    flux = uJy2galflux(uJy, transmission_params['eff_wav'], 
                    transmission_params['del_wav'], transmission)/g * t_exp * np.pi * (D*100./2)**2

    # Position of the companion
    xc = image.center.x
    yc = image.center.y
    x = companion_props['r']*rp*np.cos(companion_props['theta'])
    y = companion_props['r']*rp*np.sin(companion_props['theta'])
    x = round(xc+x)
    y = round(yc+y)

    # Sersic profile
    comp = galsim.Sersic(flux=flux*companion_props['flux_frac'], n=companion_props['n'], 
                      half_light_radius=companion_props['r_frac']*rp*pxscale, gsparams=gsparams)
    q, beta = companion_props['q'], companion_props['beta']
    t = [[np.cos(beta), -q*np.sin(beta)], [np.sin(beta), q*np.cos(beta)]]
    comp = comp.transform(t[0][0], t[0][1], t[1][0], t[1][1])/np.abs(np.linalg.det(t))

    return comp, x, y

def add_source_to_image(field, galaxy, clumps, all_xi, all_yi, 
                        comp, comp_x, comp_y, pxscale, 
                        psf_fwhm=None, psf_method='galsim',
                        use_moffat=False, **kwargs):
    """
    adding source galaxy and clumps to image *then* convolve with psf
    Args:
        image (galsim object)  : galsim image with fov and wcs set (needed for setting center)
        galaxy (galsim object) : galaxy with defined sersic profile
        clumps (list of galsim objects) : list of all clump objects to add to image
        all_xi (list of ints) : list of x positions for clumps
        all_yi (list of ints) : list of y positions for clumps
        psf_sig (float) : sigma for gaussian psf for image
        use_moffat (bool): use Moffat instead of Gaussian PSF?
    Returns:
        image_psf (galsim object) : image with psf-convolved objects added in
    """
    # make copy of image in case iterating over and changing psf each time

    if not psf_fwhm:
        psf_fwhm = 3*pxscale

    # First create the image with the galaxy (+apply swirl transform)
    image_psf = field.copy()
    if psf_method == 'galsim':

        if psf_fwhm > 0:
            # define psf
            if use_moffat:
                psf = galsim.Moffat(beta=2.45, flux=1., fwhm=psf_fwhm)
            else:
                psf = galsim.Gaussian(flux=1., sigma=psf_fwhm*gaussian_fwhm_to_sigma)

            # convolve galaxy with psf
            final_gal = galsim.Convolve([galaxy,psf])
        else:
            final_gal = galaxy

        # stamp galaxy and add to image
        stamp_gal = final_gal.drawImage(wcs=image_psf.wcs.local(image_psf.center)) #galaxy at image center
        stamp_gal.setCenter(image_psf.center.x, image_psf.center.y)
        bounds_gal = stamp_gal.bounds & image_psf.bounds
        image_psf[bounds_gal] += stamp_gal[bounds_gal]
        
        # # Swirl transform
        # gal_image = image_psf.array
        # gal_image = swirl(gal_image, strength=warp_strength, radius=gal_image.shape[0]/2)


        # Now add clumps to a blank field
        if clumps:
            for i in range(len(clumps)):
                clump = clumps[i]
                xi = all_xi[i]
                yi = all_yi[i]

                final_clump = galsim.Convolve([clump,psf]) if psf_fwhm > 0 else clump
                stamp_clump = final_clump.drawImage(wcs=image_psf.wcs.local(galsim.PositionI(xi, yi)))
                stamp_clump.setCenter(xi, yi)
                bounds_clump = stamp_clump.bounds & image_psf.bounds
                image_psf[bounds_clump] += stamp_clump[bounds_clump]

        # Add a companion
        if comp:
            final_comp = galsim.Convolve([comp,psf]) if psf_fwhm > 0 else clump
            stamp_comp = final_comp.drawImage(wcs=image_psf.wcs.local(galsim.PositionI(comp_x, comp_y)))
            stamp_comp.setCenter(comp_x, comp_y)
            bounds_comp = stamp_comp.bounds & image_psf.bounds
            image_psf[bounds_comp] += stamp_comp[bounds_comp]

        # Combine the two
        # clumps_image = image_psf.array
        # image_psf = clumps_image + gal_image
                
        # Add a tidal tail
        # if warp_strength > 0:
        # 	image_tail = field.copy()
        # 	tail = galsim.Gaussian(flux=7362179207, sigma=10)
        # 	tail = tail.shear(q=0.3, beta=-1*galsim.radians)
        # 	x = round(image_tail.center.x+r*rp*np.cos(th))
        # 	y = round(yc+r*rp*np.sin(th))
        # 	stamp_tail = tail.drawImage(wcs=image_tail.wcs.local(image_tail.center))
        # 	stamp_tail.setCenter(image_tail.center.x, image_tail.center.y)
        # 	bounds_tail = stamp_tail.bounds & image_psf.bounds
        # 	image_tail[bounds_tail] += stamp_tail[bounds_tail]
        # 	image_tail = image_tail.array
        # 	image_tail = swirl(image_tail, strength=warp_strength, radius=image_tail.shape[0])
        

        image_psf = image_psf.array 

        
    

    elif psf_method == 'astropy':
        # print('astropy')


        
        final_gal = galaxy

        # stamp galaxy and add to image
        stamp_gal = final_gal.drawImage(wcs=image_psf.wcs.local(image_psf.center)) #galaxy at image center
        stamp_gal.setCenter(image_psf.center.x, image_psf.center.y)
        bounds_gal = stamp_gal.bounds & image_psf.bounds
        image_psf[bounds_gal] += stamp_gal[bounds_gal]


        if clumps:
            for i in range(len(clumps)):
                clump = clumps[i]
                xi = all_xi[i]
                yi = all_yi[i]

                final_clump = clump
                stamp_clump = final_clump.drawImage(wcs=image_psf.wcs.local(galsim.PositionI(xi, yi)))
                stamp_clump.setCenter(xi, yi)
                bounds_clump = stamp_clump.bounds & image_psf.bounds
                image_psf[bounds_clump] += stamp_clump[bounds_clump]

        if psf_fwhm > 0:
                # define Gaussian psf
                ysize, xsize = image_psf.array.shape
                kernel = Gaussian2DKernel(x_stddev=psf_fwhm*gaussian_fwhm_to_sigma/pxscale, x_size=xsize, y_size=ysize)
                image_psf = convolve_fft(image_psf.array, kernel, boundary="wrap")
        else:
            image_psf = image_psf.array

    else:
        print('psf method not understood')

    return image_psf

def simulate_perfect_galaxy(mag, r_eff, pxscale, fov_reff=15, q=1, beta=0, n_clumps=10, 
                            n_disk=1, n_bulge=4, b_t=0.3, warp_strength=0, r_second=1,
                            prob_comp=0, comp_properties=None,
                            clump_properties=None, **kwargs):
    """Given galaxy and clump properties, simulates noiseless galaxy to a desired pixel scale.
    Args:
        mag (float): r-band magnitude
        r_eff (float): half-light radius in arcsec
        pxscale (float) pxscale in arcsec
        fov_reff (float): field-of-view in terms of effective radii
        sersic_n (float): sersic index (be careful if above 4)
        q (float): axis ratio, between 0 and 1
        beta (float): orientation in degrees
        n_clumps (int): number of asymmetry clumps
        clump_properties (dict): dict used to generate clumps, see _default_clump_properties.
    Returns:
        image_perfect (np.ndarray): the image of the ideal galaxy on a given pixel scale
        out (dict): dict containing galaxy parameters needed to make the noisy/resampled version
            'canvas' : the CCD that all models are painted on
            'galaxy' : the galaxy (no clumps) model
            'clumps' : model for all clumps
            'all_xi' : x-coordinates of all clumps
            'all_yi" : y-coordinates of all clumps
        r_pet (float): petrosian radius in arcsec
    """
    ############## Sersic profile only #########################
    # Calculate field of view in degrees
    fov = fov_reff * r_eff / 3600
    
    # generate blank image with fov and wcs info
    field_image, wcs = gen_image(sdss_ra, sdss_dec, pxscale, fov, fov)
    
    # create a galaxy with given params
    galaxy = gen_galaxy(mag=mag, re=r_eff, re_second=r_second, n_disk=n_disk, n_bulge=n_bulge, b_t=b_t, q=q, beta=beta)

    # get petrosian radius of galaxy in px
    r_pet = petrosian_sersic(fov, r_eff, 1)/pxscale
    
    ############## Asymmetry clumps ############################
    # generate all the clumps and their positions
    clumps, all_xi, all_yi = create_clumps(field_image, r_pet, mag, q, beta, clump_properties)

    ############## Companion galaxy ############################
    # If adding a companion, do it here
    if prob_comp <= companion_threshold:
        comp, comp_x, comp_y = add_companion(field_image, r_pet, mag, pxscale, comp_properties)
    else:
        comp, comp_x, comp_y = None, None, None
    
    ############## Perfect image ###############################
    image_perfect = add_source_to_image(field_image, galaxy, 
                                     clumps, all_xi, all_yi, comp, comp_x, comp_y, 
                                     pxscale, psf_fwhm=0)
    
    ############## Output array ###############################
    out = {
        'field' : field_image,
        'galaxy' : galaxy,
        'clumps' : clumps,
        'all_xi' : all_xi,
        'all_yi' : all_yi,
        'comp' : comp,
        'comp_x' : comp_x,
        'comp_y' : comp_y
    }

    # NOTE RETURNING RPET IN ARCSEC
    return np.abs(image_perfect), out, r_pet*pxscale

def get_galaxy_rng_vals(
        N, lims=_default_galaxy_properties, 
        clump_props=_default_clump_properties, comp_props=_default_companion_properties,
        alpha=1, beta=0.5, s0=1, r0=0.5, seed=None):
    
    """Generate parameters to make N perfect galaxies. No SNR or PSF effects.
    Args:
        N (int) : number of samples
        lims (dict): dictionary with min and max limits for each parameter, see _default_galaxy_properties
        clump_props (dict): same, for inidividual clumps, see _default_clump_properties
        alpha (int): scaling factor for the clump brightness as a function of Rp
        beta (int): scaling factor for the clump brightness as a function of its size 
        s0 (float): clump size relative to which normalization is applied (f ~ (s/s0)^(2-beta))
        r0 (float): radius relative to which radius norm is applied (f ~ (r/r0)^(-alpha))
        seed (int): random seed to use
    Returns:
        list[dict]: list containing N dictionaries with keys to pass to `simulate_perfect_galaxy'
    """

    # For any galaxy properties possibly not in user-supplied dict, populate with default
    for key, value in _default_galaxy_properties.items():
        if key not in lims:
            lims[key] = value
    for key, value in _default_clump_properties.items():
        if key not in clump_props:
            clump_props[key] = value

    # Using NumPy's new generator framework instead of np.random...
    rng = np.random.default_rng(seed=seed)

    #### Generate N samples with different distributions
    # Galaxy properties
    mags = rng.uniform(lims['mag'][0], lims['mag'][1], size=N)
    qs = rng.uniform(lims['q'][0], lims['q'][1], size=N)
    betas = rng.uniform(0, 2*np.pi, size=N)
    n_clumps = rng.integers(lims['n_clumps'][0], lims['n_clumps'][1]+1, size=N)
    rs = -1.9 * mags + 35 + rng.normal(lims['reff_scatter'][0], lims['reff_scatter'][1], size=N)

    # Two-sersic model
    b_t = rng.uniform(0, 1, size=N)
    n_disk = rng.uniform(lims['sersic_n'][0], 2.5, size=N)
    n_bulge = rng.uniform(2.5, lims['sersic_n'][1], size=N)
    rs2 = rng.uniform(0.1*rs, rs, size=N)

    # # Warping
    # tidal_likelihood = 0.1
    # ps_tidal = rng.uniform(0, 1, size=N)
    # warp_strength = rng.uniform(0,5,size=N)
    # warp_strength[ps_tidal >= tidal_likelihood] = 0

    # Clump properties for each galaxy
    clumps = []
    for i in range(N):
        num = n_clumps[i]
        clump_rs = rng.uniform(clump_props['r'][0], clump_props['r'][1], size=num)
        clump_fs = rng.uniform(clump_props['logf'][0], clump_props['logf'][1], size=num) #Log-uniform
        clump_size = rng.uniform(clump_props['logs'][0], clump_props['logs'][1], size=num)
        clump_theta = rng.uniform(0, 2*np.pi, size=num)
        clump_fs = np.power(10, clump_fs)
        clump_size = np.power(10, clump_size)
        
        # Rescale the clump flux based on its radius and distance
        clump_fs = clump_fs * np.power(clump_rs/r0, -alpha) * np.power(clump_size/s0, 2-beta)
        clump_dict = {
            'r' : clump_rs, 'theta' : clump_theta, 'flux_frac' : clump_fs, 'sigma' : clump_size
        }
        clumps.append(clump_dict)


    # Companion properties
    prob_comp = rng.uniform(0, 1, size=N)
    comp_properties = {}
    comp_properties['r'] = rng.uniform(comp_props['r'][0], comp_props['r'][1], size=N)
    comp_properties['theta'] = rng.uniform(0, 2*np.pi, size=N)
    comp_properties['r_frac'] = rng.uniform(comp_props['r_frac'][0], comp_props['r_frac'][1], size=N) 
    comp_properties['flux_frac'] = rng.uniform(comp_props['flux_frac'][0], comp_props['flux_frac'][1], size=N)  
    comp_properties['n'] = rng.uniform(comp_props['n'][0], comp_props['n'][1], size=N)   
    comp_properties['q'] = rng.uniform(comp_props['q'][0], comp_props['q'][1], size=N) 
    comp_properties['beta'] = rng.uniform(0, 2*np.pi, size=N)   
    comp_properties = pd.DataFrame(comp_properties).to_dict(orient="records")

    # For high-sersic index cases, set q to 1
    qs[b_t >= 0.5] = 1

    # Set lower and upper limits to the radius in arcseconds
    rs[rs < 1] = 1
    rs[rs > 20] = 20

    # Save all these values as output
    output = {
        'mag' : mags, 'r_eff' : rs, 'r_second' : rs2, 'q' : qs, 
        'b_t' : b_t, 'n_bulge' : n_bulge, 'n_disk' :n_disk, 'beta' : betas,
        'n_clumps' : n_clumps, 'clump_properties' : clumps,
        'prob_comp' : prob_comp, 'comp_properties' : comp_properties
    }

    # Convert to a list of dicts, where each dict can be passed to `simulate_perfect_galaxy`
    output = pd.DataFrame(output).to_dict(orient="records")
    return output

def get_augmentation_rng_vals(N, lims=_default_aug_properties, seed=None):
    """Generate parameters to make N observed galaxies.
    Args:
        N (int) : number of samples
        lims (dict): dictionary with min and max limits for each parameter, see _default_aug_properties
        seed (int): random seed to use
    Returns:
        list[dict]: list containing N dictionaries with keys to pass to `add_source_to_image' and 'sky_noise'.
    """
    # For any galaxy properties possibly not in user-supplied dict, populate with default
    for key, value in _default_aug_properties.items():
        if key not in lims:
            lims[key] = value

    # Using NumPy's new generator framework instead of np.random...
    rng = np.random.default_rng(seed=seed)

    # Image properties
    sky_mags = rng.uniform(lims['sky_mag'][0], lims['sky_mag'][1], size=N)
    pxscales = rng.uniform(lims['pxscale'][0], lims['pxscale'][1], size=N)

    # Save all these values as output
    output = {
        'sky_mag' : sky_mags, 'pxscale' : pxscales
    }

    # Convert to a list of dicts, where each dict can be passed to `simulate_perfect_galaxy`
    output = pd.DataFrame(output).to_dict(orient="records")
    return output


####The following is to test the code#####	

if __name__ == '__main__':

    ## transmission curve based on sdss r-band total throughput for airmass=1.3 extended source
    Filter = 'r'
    bandpass_file = "passband_sdss_" + Filter
    bandpass = galsim.Bandpass(bandpass_file, wave_type = u.angstrom)


    ## gain, exptime and diameter of telescope
    telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
    ## effective wavelength and width of filter
    transmission_params = {'eff_wav':616.5, 'del_wav':137}


    ## galaxy and sky params
    mag = 13 # mag of galaxy
    sky_mag = 22 ##mag/arcsec/arcsec sky level
    re = 10 #effective radius in arcsec
    n = 1 # sersic index
    q = 1 #axis ratio
    beta = 0 # orientation angle


    ## define ra, dec, pixel scale and fov
    centre_ra = 150
    centre_dec = 2.3
    pixel_scale = 0.4 #arcsec/pixel
    fov = re*12/3600 #deg. Basing off of re


    # generate blank image with fov and wcs info
    image, wcs = gen_image(centre_ra, centre_dec, pixel_scale, fov, fov)

    # create a galaxy with given params
    galaxy = gen_galaxy(mag=mag, re=re, n=n, q=q, beta=beta, telescope_params=telescope_params, 
        transmission_params=transmission_params, bandpass=bandpass)

    # get petrosian radius of galaxy
    rp = petrosian_sersic(fov, re, 1)/pixel_scale  ##in pixels

    # set up creation of clumps for asymmetry
    N = 20  # total number
    clump_properties = {
        'r' : [1, 2],         # Radius as a function of petrosian for each clump
        'theta' : [45, 60],   # Position angle relative to x-axis
        'flux' : [0.3, 0.1],  # Flux as a fraction of source flux
        'sigma' : [0.1, 0.9]  # Size
    }

    # positions_clumps=[(50,50)]  # positions (optional) 
    # fluxes_clumps = [0.2] # flux fractions (optional), must be same length as positions
    # sigmas_clumps = [3] # sigmas for gaussian clumps (optional), must be same length as positions

    # generate all the clumps and their positions
    clumps, all_xi, all_yi = create_clumps(
        image, rp, N, mag, telescope_params, transmission_params, bandpass, clump_properties)


    # convolve sources with psf and add to image
    image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_fwhm=2.0)

    # add Poisson noise to image based on pixel counts with added sky level
    image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)
    # FINAL IMAGE IN ELECTRON COUNTS


    ##############################################
    ## Vary the noise size, keep the psf level the same
    seed = None
    noises = [24,22,20,18]
    fig, axs = plt.subplots(nrows=1, ncols=4)
    for d in range(0,4):
        sky_mag = noises[d]


        image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_fwhm=2.0)
        image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)


        axs[d].imshow(image_noise.array, origin='lower', cmap='Greys', norm=simple_norm(image_noise.array, stretch='log', log_a=10000))
        axs[d].set_title('Sky level=' + str(sky_mag) + ' mag/arcsec^2')

    fig.show()

    ##############################################
    ## Vary the psf size, keep the noise level the same
    seed = None
    fig2, axs2 = plt.subplots(nrows=1, ncols=4)
    for d in range(1,5):

        
        image_psf = add_source_to_image(image, galaxy, clumps, all_xi, all_yi, psf_fwhm=d)
        image_noise = sky_noise(image_psf, sky_mag, pixel_scale, telescope_params, transmission_params, bandpass)


        axs2[d-1].imshow(image_noise.array, origin='lower', cmap='Greys', norm=simple_norm(image_noise.array, stretch='log', log_a=10000))
        axs2[d-1].set_title('PSF sig=' + str(d) + '"')

    fig2.show()



    input()
