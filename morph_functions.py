import numpy as np
import scipy.ndimage as ndi
from photutils import aperture
from skimage import transform as T

def get_sb_segmap(img, pxscale, lim_sb=23, zp=38):
    """Create a surface brightness contour for an image, given a SB limit.
    Returns a mask where all pixels within the given (smoothed) surface brightness threshold are 1. 
    The regions do not have to be connected; detect_sources / deblend_sources may be used later to do this.
    
    Args:
        img (np.array): the input image
        pxscale (float): the pixel scale of that image [arcsec/px]
        lim_sb (float): the desired SB limit
        zp (float): image zeropoint
    Returns:
        sb_mask (np.array): a boolean mask array where all pixels with SB over threshold are 1."""
    
    # Smooth the image with a 5x5 box to average over the noise a little
    # Use a truncated Gaussian kernel for speed
    kernel_size = 3
    img = ndi.gaussian_filter(img, sigma=kernel_size, truncate=2)

    # Calculate the contour given a SB threshold
    lim_f = pxscale**2 * np.power(10, (zp-lim_sb)/2.5)
    sb_mask = (img > lim_f).astype(float)

    # Smooth the resulting segmentation map to 1% of the image size
    # This will deal with some resolution dependence where high-res images
    # Would by default have more detail in their contours
    mask_kernel = 0.005*img.shape[0]
    sb_mask = ndi.gaussian_filter(sb_mask, mask_kernel, truncate=1)
    return sb_mask > 0.1

def get_avgsb(img, center, rpet, pxscale, ap_frac=1.5, zp=38):
    """Calculates the average SB within ap_frac * R_pet.
    
    Args:
        img (np.array): the NxN input image
        center (float, float): aperture center
        rpet (float): Petrosian radius in arcsec
        pxscale (float): image pixel scale in arcsec/px
        ap_frac (float): aperture size, in R_pet, in which average SB is calculated
        zp (float): image zeropoint
        
    Returns:
        avg_sb (float): average SB in the aperture
    """

    ap = aperture.CircularAperture(center, r=ap_frac*rpet/pxscale)
    flux = ap.do_photometry(img)[0][0]
    area = ap.area * pxscale**2
    avg_sb = -2.5*np.log10(flux/area) + zp
    return avg_sb


def shape_asymmetry(img, center, pxscale, lim_sb, mask=None, zp=38):
    """Calculate new shape asymmetry of the image, where the segmentation map is created
    using a surface brightness contour of a given SB giveb by `lim_sb`. If `lim_sb` is None,
    calculate the average SB within ap_frac * rpet aperture.

    Do not use an aperture for this calculation. This is very reliant on masking any foreground
    or background contaminants! But so would any other shape asymmetry calculation.
    
    Args:
        img (np.array): the input NxN image
        center (float, float): precomputed asymmetry center (e.g., of CAS asymmetry)
        pxscale (float): pixel scale of the image
        lim_sb (float): SB threshold used for the contour
        zp (float): image zeropoint
        mask (np.array): NxN boolean array with any masked extraneous sources
    Returns:
        shape_asym (float): shape asymmetry value
        segmap (np.array): NxN segmentation map in which shape asymmetry is calculated.
    """

    # If needed, calculate SB limit in an aperture
    # if lim_sb is None:
    #     lim_sb = get_avgsb(img, center, rpet, pxscale, ap_frac, zp)

    if mask is None:
        mask = np.zeros_like(img).astype(bool)

    # Construct the segmentation map
    # We don't deblend this -- ideally the user has made sure that unrelated sources are masked prior to this.
    # Deblending at high SB threshold would separate off SF clumps etc so it is not recommended.
    segmap = get_sb_segmap(img, pxscale, lim_sb, zp)
    segmap = segmap.astype(int)

    # Mask any extraneous sources and rotate the mask about asymmetry center
    mask_rotated = T.rotate(mask, 180, center=center, order=0)
    mask = mask.astype(bool) | mask_rotated.astype(bool)
    segmap[mask] = 0
    segmap_rotated = T.rotate(segmap, 180, center=center, order=0)

    # Calculate the shape asymmetry
    residual = np.abs(segmap - segmap_rotated)
    shape_asym = np.sum(residual)/np.sum(segmap)

    return shape_asym