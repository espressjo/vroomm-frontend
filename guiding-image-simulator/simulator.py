"""Noiseless guiding-image simulator for VROOMM frontend.

Pipeline: Gaia cone search -> tangent-plane projection -> detector position (mm)
-> PSF selection by radial zone -> per-star stamp (see `GuidingSimulator.render`)
added to the detector image.
"""
import argparse

import numpy as np
from scipy.signal import fftconvolve

from psf import ZemaxPSF

# Zemax "Center point is: row 129, column 128" (1-based) -> 0-based (row, col)
PSF_CENTER = (128, 127)


# ----------------------------------------------------------------------
def fetch_stars(ra, dec, radius_deg):
    """Gaia DR3 cone search. Returns (ra_deg, dec_deg, G_mag) arrays."""
    from astroquery.gaia import Gaia

    query = (
        "SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
        "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
        "AND phot_g_mean_mag IS NOT NULL"
    )
    t = Gaia.launch_job_async(query).get_results()
    return (np.asarray(t["ra"], float), np.asarray(t["dec"], float),
            np.asarray(t["phot_g_mean_mag"], float))


def tangent_offsets(ra, dec, ra0, dec0):
    """Gnomonic projection. Returns (xi, eta) in degrees (east, north)."""
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    dra = ra - ra0
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    xi = np.cos(dec) * np.sin(dra) / cosc
    eta = (np.cos(dec0) * np.sin(dec)
           - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cosc
    return np.degrees(xi), np.degrees(eta)


# ----------------------------------------------------------------------
def integration_matrix(edge0, step, n, j0, j1):
    """Fraction of each fine pixel's flux falling in detector pixels j0..j1-1.

    Fine pixel i spans [edge0 + i*step, edge0 + (i+1)*step] in detector-pixel
    units; detector pixel j spans [j, j+1]. Returns shape (j1 - j0, n).
    """
    a = edge0 + step * np.arange(n)
    j = np.arange(j0, j1)[:, None]
    overlap = np.clip(np.minimum(a + step, j + 1) - np.maximum(a, j), 0, None)
    return overlap / step


def gaussian_kernel(sigma_um, spacing):
    """Normalized (sum = 1) 2D Gaussian sampled at `spacing`, truncated at 4 sigma."""
    h = max(int(np.ceil(4 * sigma_um / spacing)), 1)
    x = np.arange(-h, h + 1) * spacing
    g = np.exp(-0.5 * (x / sigma_um) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


class GuidingSimulator:
    """
    Parameters
    ----------
    psf_files : list of str
        Zemax PSF files, ordered from the optical axis outwards.
    shape : (ny, nx)
        Detector size in pixels.
    pixel_size : float
        Detector pixel pitch in µm.
    seeing : float
        Atmospheric seeing FWHM in arcsec (Gaussian atmospheric PSF).
    flux_scale : float
        Total detector counts of the brightest star in the queried field; all
        other stars are scaled relative to it (flux ratio 10**(-0.4*dmag)).
    """

    def __init__(self, psf_files, shape=(1216, 1936), pixel_size=5.86,
                 seeing=1.5, flux_scale=1e5):
        self.psfs = [ZemaxPSF(f) for f in psf_files]
        self.shape = shape
        self.pixel_size = pixel_size
        self.seeing = seeing
        self.flux_scale = flux_scale

        spacings = {p.scale for p in self.psfs}
        if len(spacings) != 1:
            raise ValueError(f"PSFs have different spacings: {spacings}")
        self.spacing = spacings.pop()

        # Radius on the detector (mm) of each PSF, and its field angle (deg).
        self.psf_radius_mm = np.array([np.hypot(*p.reference_coordinate)
                                       for p in self.psfs])
        self.psf_field = np.array([np.hypot(*p.field) for p in self.psfs])
        if np.any(np.diff(self.psf_radius_mm) <= 0):
            raise ValueError("PSFs must be ordered by increasing radius")

        # Steps 1-3 are identical for every star in a zone: do them once.
        self.effective_psfs = [self._effective_psf(i)
                               for i in range(len(self.psfs))]

    # ------------------------------------------------------------------
    def _effective_psf(self, i):
        """Optical PSF (x) atmospheric PSF, normalized to unit sum."""
        # 1. Normalize optical PSF
        optical = self.psfs[i].data / self.psfs[i].data.sum()
        # 2. Convolve with normalized atmospheric PSF (Gaussian, FWHM = seeing).
        #    Plate scale (arcsec/µm) is taken from the nearest off-axis PSF.
        j = max(i, 1)
        arcsec_per_um = self.psf_field[j] * 3600 / (self.psf_radius_mm[j] * 1e3)
        sigma_um = self.seeing / arcsec_per_um / (2 * np.sqrt(2 * np.log(2)))
        if sigma_um <= 0:
            return optical
        atmos = gaussian_kernel(sigma_um, self.spacing)
        effective = fftconvolve(optical, atmos, mode="same")
        # 3. Normalize effective PSF (restores the flux lost to the stamp edge)
        return effective / effective.sum()

    # ------------------------------------------------------------------
    def _extrap(self, x, xp, fp):
        """np.interp with linear extrapolation beyond the last point."""
        y = np.interp(x, xp, fp)
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), y)

    def field_radius_deg(self):
        """Field angle (deg) at the detector corner."""
        ny, nx = self.shape
        r_mm = 0.5 * np.hypot(nx, ny) * self.pixel_size / 1e3
        return float(self._extrap(r_mm, self.psf_radius_mm, self.psf_field))

    def sky_to_detector_mm(self, xi, eta):
        """Tangent-plane offsets (deg) -> detector (x, y) in mm.

        The radial field-angle -> mm relation is interpolated from the PSF
        reference coordinates (origin at the PSF centre (0, 0)). North is up
        (+y), east is left (-x).
        """
        r_deg = np.hypot(xi, eta)
        r_mm = self._extrap(r_deg, np.r_[0.0, self.psf_field[1:]],
                            np.r_[0.0, self.psf_radius_mm[1:]])
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(r_deg > 0, r_mm / r_deg, 0.0)
        return -xi * k, eta * k

    def psf_index(self, x_mm, y_mm):
        """Zone i = [r_i, r_{i+1}) uses PSF i; beyond the last uses the last."""
        r = np.hypot(x_mm, y_mm)
        return np.clip(np.searchsorted(self.psf_radius_mm, r, side="right") - 1,
                       0, len(self.psfs) - 1)

    # ------------------------------------------------------------------
    def render(self, x_mm, y_mm, mag):
        """Render stars given by detector position (mm) and G magnitude.

        Per star: pick the zone's effective PSF (normalized, steps 1-3 done in
        `_effective_psf`), then
          4. scale by the star flux,
          5. shift to the star's subpixel position,
          6. integrate over detector pixels,
          7. add to the detector image.
        """
        ny, nx = self.shape
        pix = self.pixel_size
        step = self.spacing / pix            # fine pixel size in detector pixels
        img = np.zeros(self.shape)
        x_mm, y_mm = np.asarray(x_mm, float), np.asarray(y_mm, float)
        rel = 10 ** (-0.4 * (np.asarray(mag, float) - np.min(mag)))
        flux = self.flux_scale * rel
        # Star positions in detector-pixel units (pixel j spans [j, j+1])
        px = nx / 2 + x_mm * 1e3 / pix
        py = ny / 2 + y_mm * 1e3 / pix
        idx = self.psf_index(x_mm, y_mm)
        cy, cx = PSF_CENTER

        for sx, sy, f, i in zip(px, py, flux, idx):          # 8. repeat
            psf = self.effective_psfs[i]
            kh, kw = psf.shape
            # 4. scale by star flux
            stamp = f * psf
            # 5. shift to subpixel position: the stamp's centre pixel is
            #    centred on (sx, sy), so fine-pixel edges start here
            ex = sx + (-cx - 0.5) * step
            ey = sy + (-cy - 0.5) * step
            # 6. integrate over detector pixels (exact box overlap, valid for
            #    any pixel-size / PSF-sampling ratio), clipped to the detector
            xa, xb = max(int(np.floor(ex)), 0), min(int(np.ceil(ex + kw * step)), nx)
            ya, yb = max(int(np.floor(ey)), 0), min(int(np.ceil(ey + kh * step)), ny)
            if xa >= xb or ya >= yb:
                continue
            Mx = integration_matrix(ex, step, kw, xa, xb)
            My = integration_matrix(ey, step, kh, ya, yb)
            # 7. add to detector image
            img[ya:yb, xa:xb] += My @ stamp @ Mx.T
        return img

    def simulate(self, ra, dec, margin=1.05):
        """Download the Gaia field around (ra, dec) [deg] and render it."""
        sr, sd, mag = fetch_stars(ra, dec, self.field_radius_deg() * margin)
        xi, eta = tangent_offsets(sr, sd, ra, dec)
        x_mm, y_mm = self.sky_to_detector_mm(xi, eta)
        return self.render(x_mm, y_mm, mag)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ra", type=float, help="RA of detector centre (deg)")
    ap.add_argument("dec", type=float, help="Dec of detector centre (deg)")
    ap.add_argument("--psf", nargs="+",
                    default=[f"psf/psf-{i}.txt" for i in range(4)])
    ap.add_argument("--size", type=int, nargs=2, default=(1216, 1936),
                    metavar=("NY", "NX"))
    ap.add_argument("--pixel-size", type=float, default=5.86, help="µm")
    ap.add_argument("--seeing", type=float, default=1.5, help="FWHM, arcsec")
    ap.add_argument("--flux-scale", type=float, default=1e5,
                    help="counts of the brightest star in the field")
    ap.add_argument("-o", "--output", default="sim.fits")
    args = ap.parse_args()

    sim = GuidingSimulator(args.psf, tuple(args.size), args.pixel_size,
                           args.seeing, args.flux_scale)
    image = sim.simulate(args.ra, args.dec)

    from astropy.io import fits
    hdr = fits.Header()
    hdr["CRVAL1"], hdr["CRVAL2"] = args.ra, args.dec
    hdr["PIXSIZE"] = (args.pixel_size, "um")
    hdr["SEEING"] = (args.seeing, "arcsec FWHM")
    hdr["FLUXSCAL"] = (args.flux_scale, "counts of brightest star")
    fits.PrimaryHDU(image.astype("float32"), hdr).writeto(args.output,
                                                          overwrite=True)
    print(f"wrote {args.output}: sum={image.sum():.4g} max={image.max():.4g}")
