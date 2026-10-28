"""Simulate a noisy guiding-camera star field for the VROOMM frontend.

Pipeline (see simulated-images.md for the math):
  Gaia DR3 cone search (G_RP) -> tangent-plane projection -> detector position
  (mm) -> G_RP magnitude to electrons (1.6 m telescope x guiding-branch
  throughput x GT1930 QE) -> optical PSF (x) seeing -> integration on detector
  pixels -> Poisson / dark / read noise.

Example:
    python simulated_images.py 11.8 85.24 -t 10 --seeing 1.5 --temperature -5 \\
        -o sim.fits
"""
import argparse

import numpy as np
from scipy.signal import fftconvolve

from detector import Detector
from psf import PSFSet
from throughput import guiding

# Gaia DR3 G_RP passband (SVO filter GAIA/GAIA3.Grp), Vega system
RP_ZEROPOINT = 1.26902e-9            # erg / cm^2 / s / Angstrom
RP_LAMBDA_MIN, RP_LAMBDA_MAX = 0.620046, 1.046557   # µm

_trapz = getattr(np, "trapezoid", None) or np.trapz
HC = 1.98645e-12                     # erg * µm  (Planck * c)
TELESCOPE_DIAMETER = 1.6             # m
OBSTRUCTION_RADIUS = 0.2             # m, central obstruction radius
SKY_BRIGHTNESS = 20.5                # G_RP mag / arcsec^2 (dark-site guess)


# ----------------------------------------------------------------------
# Gaia + astrometry
def fetch_stars(ra, dec, radius_deg):
    """Gaia DR3 cone search. Returns (ra_deg, dec_deg, G_RP mag) arrays."""
    from astroquery.gaia import Gaia

    query = (
        "SELECT ra, dec, phot_rp_mean_mag FROM gaiadr3.gaia_source "
        "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) "
        "AND phot_rp_mean_mag IS NOT NULL"
    )
    t = Gaia.launch_job_async(query).get_results()
    return (np.asarray(t["ra"], float), np.asarray(t["dec"], float),
            np.asarray(t["phot_rp_mean_mag"], float))


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
# Rendering helpers
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


# ----------------------------------------------------------------------
class StarFieldSimulator:
    """Noisy star-field generator.

    Parameters
    ----------
    psfs : PSFSet, optional
    detector : Detector, optional
    throughput : guiding, optional
        Guiding-branch throughput (telescope mirrors included).
    seeing : float
        Seeing FWHM in arcsec.
    diameter : float
        Telescope primary diameter in metres.
    obstruction_radius : float
        Central obstruction radius in metres (default 0.2).
    sky_brightness : float or None
        Sky surface brightness in G_RP mag/arcsec^2; None for no sky.
    """

    def __init__(self, psfs=None, detector=None, throughput=None, seeing=1.5,
                 diameter=TELESCOPE_DIAMETER,
                 obstruction_radius=OBSTRUCTION_RADIUS,
                 sky_brightness=SKY_BRIGHTNESS):
        self.psfs = psfs or PSFSet()
        self.detector = detector or Detector()
        self.throughput = throughput or guiding()
        self.seeing = seeing
        self.sky_brightness = sky_brightness
        self.area_cm2 = np.pi * ((diameter / 2) ** 2 - obstruction_radius ** 2) * 1e4

        spacings = {p.scale for p in self.psfs}
        if len(spacings) != 1:
            raise ValueError(f"PSFs have different spacings: {spacings}")
        self.spacing = spacings.pop()                      # µm per PSF pixel

        # Field angle (deg) -> detector radius (mm), from the PSF reference
        # coordinates, with the optical axis as origin.
        r_deg = np.array([np.hypot(*p.field) for p in self.psfs])
        r_mm = self.psfs.radius_mm
        r_deg, i = np.unique(np.r_[0.0, r_deg], return_index=True)
        self._r_deg, self._r_mm = r_deg, np.r_[0.0, r_mm][i]
        if len(self._r_deg) < 2:
            raise ValueError("Need at least one off-axis PSF to get a plate scale")

        self.photon_integral = self._photon_integral()
        self._effective = {}                               # PSF index -> stamp

    # ------------------------------------------------------------------
    # Photometry
    def _photon_integral(self):
        """I = ∫ T(λ) QE(λ) λ/(hc) dλ over the G_RP window, in photons/erg·µm."""
        wl = np.linspace(RP_LAMBDA_MIN, RP_LAMBDA_MAX, 2000)
        y = self.throughput(wl) * self.detector.qe(wl) * wl / HC
        return float(_trapz(y, wl))

    def electron_rate(self, rp_mag):
        """Electrons per second collected from a star of G_RP magnitude `rp_mag`.

        Flat f_λ spectrum equal to the Vega zero point scaled by the magnitude.
        """
        f_lambda = RP_ZEROPOINT * 1e4 * 10 ** (-0.4 * np.asarray(rp_mag, float))
        return f_lambda * self.area_cm2 * self.photon_integral   # erg/s/cm²/µm

    def sky_electrons(self, t_int):
        """Mean sky electrons per detector pixel in `t_int` seconds.

        A surface brightness of m mag/arcsec^2 puts the flux of an m-mag star
        in every arcsec^2, so e-/pixel = R(m) * (pixel solid angle) * t.
        """
        if self.sky_brightness is None:
            return 0.0
        pix_arcsec = self.detector.pixel_size * self.plate_scale(0)
        return float(self.electron_rate(self.sky_brightness)
                     * pix_arcsec ** 2 * t_int)

    # ------------------------------------------------------------------
    # Geometry
    @staticmethod
    def _extrap(x, xp, fp):
        """np.interp with linear extrapolation beyond the last point."""
        y = np.interp(x, xp, fp)
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), y)

    def field_radius_deg(self):
        """Field angle (deg) at the detector corner."""
        ny, nx = self.detector.shape
        r_mm = 0.5 * np.hypot(nx, ny) * self.detector.pixel_size / 1e3
        return float(self._extrap(r_mm, self._r_mm, self._r_deg))

    def sky_to_detector_mm(self, xi, eta):
        """Tangent-plane offsets (deg) -> detector (x, y) mm. North up, east left."""
        r_deg = np.hypot(xi, eta)
        r_mm = self._extrap(r_deg, self._r_deg, self._r_mm)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(r_deg > 0, r_mm / r_deg, 0.0)
        return -xi * k, eta * k

    def plate_scale(self, i):
        """Plate scale (arcsec/µm) of PSF `i` (nearest off-axis PSF if on-axis)."""
        r_mm = self.psfs.radius_mm[i]
        if r_mm <= 0:
            return self._r_deg[1] * 3600 / (self._r_mm[1] * 1e3)
        return np.hypot(*self.psfs[i].field) * 3600 / (r_mm * 1e3)

    # ------------------------------------------------------------------
    # PSF
    def effective_psf(self, i):
        """Optical PSF `i` (x) seeing, normalized to unit sum (cached)."""
        if i not in self._effective:
            optical = self.psfs[i].data / self.psfs[i].data.sum()
            sigma_um = (self.seeing / self.plate_scale(i)
                        / (2 * np.sqrt(2 * np.log(2))))
            if sigma_um > 0:
                optical = fftconvolve(
                    optical, gaussian_kernel(sigma_um, self.spacing), mode="same")
            self._effective[i] = optical / optical.sum()
        return self._effective[i]

    # ------------------------------------------------------------------
    def render(self, x_mm, y_mm, electrons):
        """Noiseless image (electrons) of stars with total `electrons` each."""
        ny, nx = self.detector.shape
        pix = self.detector.pixel_size
        step = self.spacing / pix            # fine pixel size in detector pixels
        img = np.zeros((ny, nx))
        px = nx / 2 + np.asarray(x_mm, float) * 1e3 / pix
        py = ny / 2 + np.asarray(y_mm, float) * 1e3 / pix

        for sx, sy, x, y, e in zip(px, py, x_mm, y_mm, electrons):
            psf = self.effective_psf(self.psfs.index(x, y))
            kh, kw = psf.shape
            # Zemax centre point (1-based) is row N/2+1, column N/2
            cy, cx = kh // 2, kw // 2 - 1
            ex = sx + (-cx - 0.5) * step
            ey = sy + (-cy - 0.5) * step
            xa, xb = max(int(np.floor(ex)), 0), min(int(np.ceil(ex + kw * step)), nx)
            ya, yb = max(int(np.floor(ey)), 0), min(int(np.ceil(ey + kh * step)), ny)
            if xa >= xb or ya >= yb:
                continue
            Mx = integration_matrix(ex, step, kw, xa, xb)
            My = integration_matrix(ey, step, kh, ya, yb)
            img[ya:yb, xa:xb] += My @ (e * psf) @ Mx.T
        return img

    def simulate(self, ra, dec, t_int, temperature=0.0, mag_limit=None,
                 rng=None, margin=1.05):
        """Download Gaia around (ra, dec) [deg] and return the noisy image (ADU)."""
        sr, sd, mag = fetch_stars(ra, dec, self.field_radius_deg() * margin)
        if mag_limit is not None:
            keep = mag <= mag_limit
            sr, sd, mag = sr[keep], sd[keep], mag[keep]
        xi, eta = tangent_offsets(sr, sd, ra, dec)
        x_mm, y_mm = self.sky_to_detector_mm(xi, eta)
        self.n_stars = len(mag)
        signal = self.render(x_mm, y_mm, self.electron_rate(mag) * t_int)
        self.signal = signal
        self.sky = self.sky_electrons(t_int)
        return self.detector.noisy_image(t_int, temperature, signal=signal,
                                         sky=self.sky, rng=rng)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ra", type=float, help="RA of detector centre (deg)")
    ap.add_argument("dec", type=float, help="Dec of detector centre (deg)")
    ap.add_argument("-t", "--exposure", type=float, required=True,
                    help="exposure time (s), 0 < t <= 30")
    ap.add_argument("--seeing", type=float, default=1.5,
                    help="seeing FWHM, arcsec (default 1.5)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="CCD temperature, deg C (default 0)")
    ap.add_argument("--sky-brightness", type=float, default=SKY_BRIGHTNESS,
                    help="sky surface brightness, G_RP mag/arcsec^2 "
                         f"(default {SKY_BRIGHTNESS})")
    ap.add_argument("--no-sky", action="store_true", help="disable the sky")
    ap.add_argument("--mag-limit", type=float, default=None,
                    help="ignore stars fainter than this G_RP magnitude")
    ap.add_argument("--diameter", type=float, default=TELESCOPE_DIAMETER,
                    help="telescope diameter, m (default 1.6)")
    ap.add_argument("--obstruction-radius", type=float, default=200.0,
                    help="central obstruction radius, mm (default 200)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("-o", "--output", default="simulated.fits",
                    help="output FITS file (default simulated.fits)")
    ap.add_argument("--show", action="store_true", help="display the image")
    args = ap.parse_args()

    sim = StarFieldSimulator(seeing=args.seeing, diameter=args.diameter,
                             obstruction_radius=args.obstruction_radius / 1e3,
                             sky_brightness=None if args.no_sky
                             else args.sky_brightness)
    image = sim.simulate(args.ra, args.dec, args.exposure, args.temperature,
                         args.mag_limit, args.seed)

    from astropy.io import fits
    hdr = fits.Header()
    hdr["CRVAL1"], hdr["CRVAL2"] = args.ra, args.dec
    hdr["RA"], hdr["DEC"] = args.ra, args.dec
    hdr["EXPTIME"] = (args.exposure, "s")
    hdr["SEEING"] = (args.seeing, "arcsec FWHM")
    hdr["CCDTEMP"] = (args.temperature, "deg C")
    hdr["SKYMAG"] = (-1 if args.no_sky else args.sky_brightness,
                     "G_RP mag/arcsec2 (-1 = no sky)")
    hdr["SKYE"] = (sim.sky, "mean sky e-/pixel over exposure")
    hdr["GAIN"] = (sim.detector.gain, "e-/ADU")
    hdr["RDNOISE"] = (sim.detector.read_noise, "e- rms")
    hdr["PIXSIZE"] = (sim.detector.pixel_size, "um")
    hdr["NSTARS"] = (sim.n_stars, "Gaia stars rendered")
    fits.PrimaryHDU(image.astype("float32"), hdr).writeto(args.output,
                                                          overwrite=True)
    print(f"wrote {args.output}: {sim.n_stars} stars, "
          f"max={image.max():.0f} ADU, median={np.median(image):.1f} ADU")
    if args.show:
        sim.detector.show(image)
