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

# New morphology functions
import morph_functions as new_morph

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()






def get_segmap(image):
    std = sigma_clipped_stats(image)[2]
    kernel = Gaussian2DKernel(5)
    img_smooth = convolve(image, kernel)
    segmap = detect_sources(img_smooth, std, 100)

    # Keep only the central region
    areas = segmap.areas
    maxlabel = np.argmax(areas)+1
    s_arr = segmap.data
    mask = (s_arr > 0) & (s_arr != maxlabel)
    segmap.keep_label(maxlabel)
    segmap.relabel_consecutive()

    return segmap, mask

def get_weightmap(image):
    std = sigma_clipped_stats(image)[2]
    noisemap = np.sqrt(image + std**2)
    return noisemap


def get_sm_output(source_morph):
    # Statmorph output we want to keep
    sm_keys = [
        'flag', 'flag_sersic',  'xc_centroid', 'yc_centroid', 'ellipticity_centroid', 'elongation_centroid', 
        'orientation_centroid', '_sky_asymmetry', 'xc_asymmetry', 'yc_asymmetry', 'ellipticity_asymmetry', 
        'elongation_asymmetry', 'orientation_asymmetry', 'rpetro_circ', 'flux_circ', 'rpetro_ellip', 'flux_ellip', 
        'rmax_circ', 'rmax_ellip', 'rhalf_circ', 'rhalf_ellip', 'r20', 'r50', 'r80', 'gini', 'm20',
        'gini_m20_bulge', 'gini_m20_merger', 'sn_per_pixel', 'concentration', 'asymmetry', 
        'smoothness', 'multimode', 'intensity', 'deviation', 'outer_asymmetry', 'shape_asymmetry', 
        'sersic_amplitude', 'sersic_rhalf', 'sersic_n', 'sersic_xc', 'sersic_yc', 
        'sersic_ellip', 'sersic_theta', 'sersic_chi2_dof', 'sersic_bic']
    
    sm_vals = source_morph.__dict__
    out_dict = {key: value for key, value in sm_vals.items() if key in sm_keys}
    return out_dict


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
    masks = []
    for img in imgs:
        segmap, mask = get_segmap(img)
        segmaps.append(segmap)
        masks.append(mask)
    
    # Weightmaps
    weightmaps = []
    for img, sky in zip(imgs, [base_sky, sky_flux]):
        noise = np.sqrt(img + sky**2)
        weightmaps.append(noise)
    
    # Run statmorph
    morphs = []
    for img, segmap, weight, mask in zip(imgs, segmaps, weightmaps, masks):
        morph = statmorph.source_morphology(img, segmap, weightmap=weight, 
                                            mask=mask, psf=psf, include_doublesersic=True)[0]
        morphs.append(morph)


    ####### Shape asymmetry add-on
    # Avg SB
    pxscales = [perfect_pxscale, aug_params['pxscale']]
    ashape_dicts = [{}, {}] # Base, obs
    for i, img in enumerate(imgs):
        cent = morphs[i].xc_asymmetry, morphs[i].yc_asymmetry
        rpet = morphs[i].rpetro_circ

        # Shape asymmetry at 1.5Rp average SB
        sb_avg = new_morph.get_avgsb(img, cent, rpet*pxscales[i], pxscales[i]) 
        ashape_avg = new_morph.shape_asymmetry(img, cent, pxscales[i], sb_avg, masks[i])
        ashape_dicts[i][f'avg_sb'] = sb_avg 
        ashape_dicts[i][f'ashape_avg'] = ashape_avg

        sb_lims = np.arange(20,28)
        for lim in sb_lims:
            ashape = new_morph.shape_asymmetry(img, cent, pxscales[i], lim, masks[i])
            ashape_dicts[i][f'ashape_{lim}'] = ashape
    

    # Save the output
    outdict = {
        'morph_base' : get_sm_output(morphs[0]),
        'morph_obs' : get_sm_output(morphs[1]),
        'ashape_base' : ashape_dicts[0],
        'ashape_obs' : ashape_dicts[1],
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
    # for p in aug_params:
    #     p['pxscale'] = perfect_pxscale
#         p['sky_mag'] = 30

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
           filepath=f'{args.path}/{i}.pkl', gal_params=gal_params[i], 
           aug_params=aug_params[i], perfect_pxscale=perfect_pxscale
    ) for i in tqdm(range(N), total=N) )


