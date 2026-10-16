"""Add detector and sky noise to a noiseless simulated image (FITS in, FITS out).

The input image is assumed to be in ADU. Noise terms are generated in electrons
and converted to ADU with the gain before being added:
  - dark current: Poisson(dark_current * t_int)
  - sky background: Poisson(sky)
  - readout noise: Gaussian(0, read_noise)
Photon noise of the stars themselves is not added.
"""
import argparse

import numpy as np
from astropy.io import fits


def add_noise(image, t_int, sky=0.0, gain=8.0, read_noise=4.5, dark_current=4.0,
              rng=None):
    """Return `image` (ADU) plus dark, sky and readout noise (all in ADU).

    Parameters
    ----------
    image : np.ndarray
        Noiseless image in ADU.
    t_int : float
        Integration time in seconds.
    sky : float
        Mean sky background in electrons per pixel over the exposure.
    gain : float
        Electrons per ADU.
    read_noise : float
        Readout noise, electrons rms.
    dark_current : float
        Electrons per pixel per second.
    """
    rng = np.random.default_rng(rng)
    dark = rng.poisson(dark_current * t_int, image.shape)
    background = rng.poisson(sky, image.shape)
    readout = rng.normal(0.0, read_noise, image.shape)
    return image + (dark + background + readout) / gain


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="noiseless FITS image (ADU)")
    ap.add_argument("-t", "--t-int", type=float, required=True,
                    help="integration time (s)")
    ap.add_argument("--sky", type=float, default=0.0,
                    help="sky background, electrons per pixel over the exposure")
    ap.add_argument("--gain", type=float, default=8.0, help="e-/ADU")
    ap.add_argument("--read-noise", type=float, default=4.5, help="e- rms")
    ap.add_argument("--dark-current", type=float, default=4.0, help="e-/s/pixel")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output", default=None,
                    help="output FITS (default: <input>_noisy.fits)")
    args = ap.parse_args()

    with fits.open(args.input) as hdul:
        image = hdul[0].data.astype(float)
        hdr = hdul[0].header.copy()

    noisy = add_noise(image, args.t_int, args.sky, args.gain, args.read_noise,
                      args.dark_current, args.seed)

    hdr["EXPTIME"] = (args.t_int, "s")
    hdr["GAIN"] = (args.gain, "e-/ADU")
    hdr["RDNOISE"] = (args.read_noise, "e- rms")
    hdr["DARKCUR"] = (args.dark_current, "e-/s/pixel")
    hdr["SKYLEVEL"] = (args.sky, "e-/pixel over exposure")
    out = args.output or args.input.rsplit(".fits", 1)[0] + "_noisy.fits"
    fits.PrimaryHDU(noisy.astype("float32"), hdr).writeto(out, overwrite=True)
    print(f"wrote {out}: mean={noisy.mean():.3f} std={noisy.std():.3f} ADU")
