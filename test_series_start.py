import numpy as np
from tqdm import tqdm
import argparse
import pickle
import statmorph

# Astropy
from photutils.segmentation import detect_sources
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve, Tophat2DKernel

# Galsim
from galaxy_generator import (
    get_galaxy_rng_vals, get_augmentation_rng_vals,
    simulate_perfect_galaxy, add_source_to_image, sky_noise
)

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()


def get_segmap(image):
    std = sigma_clipped_stats(image)[2]
    kernel = Gaussian2DKernel(5)
    img_smooth = convolve(image, kernel)
    segmap = detect_sources(img_smooth, std, 100)
    return segmap

def get_weightmap(image):
    std = sigma_clipped_stats(image)[2]
    noisemap = np.sqrt(image + std**2)
    return noisemap


def single_galaxy_run(filepath, gal_params, aug_params, perfect_pxscale):

    # Perfect image
    gal_perfect, gal_dict, rpet = simulate_perfect_galaxy(pxscale=perfect_pxscale, **gal_params)
    gal_base = add_source_to_image(**gal_dict, pxscale=perfect_pxscale) # Apply PSF
    gal_base, base_sky = sky_noise(gal_base, 30, perfect_pxscale)

    # Observed image
    gal_lowres, gal_dict, _ = simulate_perfect_galaxy(**gal_params, **aug_params)
    gal_psf = add_source_to_image(**gal_dict, **aug_params)
    gal_obs, sky_flux = sky_noise(gal_psf, **aug_params)
    imgs = [gal_base, gal_obs]

    # PSF is just three pixels wide
    psf = Gaussian2DKernel(3*gaussian_fwhm_to_sigma).array

    # Segmaps
    segmaps = []
    for img in imgs:
        segmap = get_segmap(img)
        segmaps.append(segmap)
    
    # Weightmaps
    weightmaps = []
    for img, sky in zip(imgs, [base_sky, sky_flux]):
        noise = np.sqrt(img + sky**2)
        weightmaps.append(noise)
    
    # Run statmorph
    morphs = []
    for img, segmap, weight in zip(imgs, segmaps, weightmaps):
        morph = statmorph.source_morphology(img, segmap, weightmap=weight, psf=psf)
        morphs.append(morph)

    # Save the output
    outdict = {
        'morph_base' : morphs[0],
        'morph_obs' : morphs[1],
        'gal_params' : gal_params,
        'aug_params' : aug_params,
        'perfect_pxscale' : perfect_pxscale
    }
    with open(filepath, 'wb') as f:
        pickle.dump(outdict, f)
    

if __name__ == '__main__':

    ###### Parallelize over different galaxies. For each, do a PSF and SNR series.
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of galaxies to generate")
    parser.add_argument("path", help="folder to store images and asymmetries in")
    args = parser.parse_args()

    # Perfect resolution pxscale
    perfect_pxscale = 0.1
    
    # Generate random params
    N = int(args.N)
    gal_params = get_galaxy_rng_vals(N)
    aug_params = get_augmentation_rng_vals(N)
    
    # Fix the parameters other than the one I want to vary
    for p in aug_params:
        p['pxscale'] = perfect_pxscale

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
           filepath=f'{args.path}/{i}.pkl', gal_params=gal_params[i], 
           aug_params=aug_params[i], perfect_pxscale=perfect_pxscale
    ) for i in tqdm(range(N), total=N) )


