# Simulated guiding images

`simulated_images.py` produces a noisy guiding-camera image of a real star field.
This document describes the physics and maths, then the command line.

Modules used: `psf.py` (`PSFSet`), `detector.py` (`Detector`), `throughput.py`
(`guiding`).

## 1. Pipeline

1. Cone search of Gaia DR3 around the requested (RA, Dec): positions and `phot_rp_mean_mag` (G_RP).
2. Project the stars onto the tangent plane and convert to detector position (mm).
3. Convert each G_RP magnitude to a number of electrons collected in the exposure.
4. Build the PSF at each star's location: Zemax optical PSF convolved with the seeing.
5. Place each PSF at its sub-pixel position and integrate over detector pixels.
6. Add noise (photon, dark, readout), convert to ADU, quantize, saturate.

## 2. From G_RP magnitude to electrons

### 2.1 Flux density from the magnitude

Gaia magnitudes are in the Vega system. For the G_RP passband (SVO filter `GAIA/GAIA3.Grp`) the Vega zero-point flux density is

    f0 = 1.26902e-9 erg s^-1 cm^-2 Å^-1   (λ_pivot = 0.7769 µm)

A star of magnitude m has

    f_λ(m) = f0 · 10^(-0.4 m)

We assume a **flat spectrum** (constant f_λ) across the G_RP window,
λ = 0.620 – 1.047 µm (SVO λ_min, λ_max). This follows the stated assumption that G_RP covers a wavelength range similar to the instrument's. No stellar colour correction is applied.

### 2.2 Photons

A photon of wavelength λ carries E = hc/λ, with hc = 1.98645e-12 erg·µm.
The photon flux density is therefore f_λ · λ / (hc) photons s^-1 cm^-2 µm^-1.
Converting f_λ from per Å to per µm multiplies by 10^4.

### 2.3 Electrons

With the telescope collecting area A and the end-to-end efficiency
T(λ) · QE(λ), the electron rate is

    R(m) = A · f_λ(m) · I           [e- / s]

    I = ∫ T(λ) · QE(λ) · λ / (hc) dλ     over 0.620 – 1.047 µm

- A = π [(D/2)² − r_obs²], with D = 1.6 m and a central obstruction of radius r_obs = 200 mm (defaults), so A = 1.885e4 cm².
- T(λ) is `guiding` from `throughput.py`. It is the full guiding-branch throughput **including the telescope mirrors**, so no extra mirror reflectance is applied.
- QE(λ) is `Detector.qe` from the GT1930 curve (`references/gt1930-qe.csv`).
- I is evaluated once, by trapezoidal integration on a 2000-point grid. Both T and QE are 0 outside their tabulated ranges, so the integral stops at 1.0 µm, the end of the throughput table. This slightly underestimates the flux of the red tail (1.0 – 1.047 µm).

Because f_λ ∝ 10^(-0.4 m), all stars share the same I and the rate is

    R(m) = R(0) · 10^(-0.4 m)

The total electrons in an exposure of t seconds are N = R(m) · t. As a reference, a G_RP = 10 star gives about 1.6e4 e-/s with the current throughput file (which is ~2 % across the band, see `throughput.py`).

Gaia's own electron count is *not* used: it is tied to Gaia's pupil, optics and CCDs. We only use the magnitude as a calibrated flux.

### 2.4 Sky background

The sky is approximated as a uniform surface brightness μ (G_RP mag/arcsec², default 20.5, a dark-site guess). One arcsec² of sky carries the flux of a magnitude-μ star, so the mean sky electrons per pixel are

    N_sky = R(μ) · Ω_pix · t ,   Ω_pix = (pixel_size · s)²  arcsec²

where s is the on-axis plate scale (arcsec/µm), about 0.175″/pixel for a 5.86 µm pixel. With the current throughput this is about 0.03 e-/pixel/s. Seeing does not change a uniform sky. The sky enters the Poisson draw of section 5, so its photon noise is included. `--no-sky` turns it off.

## 3. Geometry

### 3.1 Tangent plane

Stars are projected about the field centre (RA₀, Dec₀) with the gnomonic projection:

    cos c = sin δ₀ sin δ + cos δ₀ cos δ cos(α − α₀)
    ξ = cos δ sin(α − α₀) / cos c
    η = (cos δ₀ sin δ − sin δ₀ cos δ cos(α − α₀)) / cos c

(ξ east, η north, in degrees.)

### 3.2 Plate scale and distortion

Each Zemax PSF gives a pair (field angle θ, detector radius r). Together with the origin (0, 0) they define a radial map r(θ), linearly interpolated and linearly extrapolated beyond the outermost PSF (the detector corners lie beyond it). The detector position is

    x = −ξ · r(θ)/θ ,   y = η · r(θ)/θ        (north up, east left)

with θ = √(ξ² + η²). The local plate scale of PSF i is s_i = 3600 · θ_i / (1000 · r_i) arcsec/µm. The on-axis PSF uses the plate scale of the nearest off-axis PSF. Pixel coordinates are

    p_x = N_x/2 + x/pixel_size ,   p_y = N_y/2 + y/pixel_size

The Gaia search radius is the field angle at the detector corner (× 1.05).

## 4. PSF

### 4.1 Selection

`PSFSet.get(x, y)` picks the PSF for the star location. Along one direction, PSF i is used for r in [r_i, r_{i+1}), and PSFs at different position angles are selected by nearest direction (see `psf.py`).

### 4.2 Seeing

The atmosphere is a normalized 2D Gaussian with FWHM = seeing. It is converted to µm on the detector with the plate scale of the selected PSF:

    σ[µm] = seeing / s_i / (2 √(2 ln 2))

The effective PSF is

    PSF_eff = normalize( PSF_optical ⊛ Gaussian )

Normalizing to unit sum puts back the flux that the finite stamp would otherwise lose. The effective PSFs are cached per PSF index, because they don't depend on the star.

### 4.3 Sub-pixel placement and integration

The PSF is sampled at 0.5 µm, much finer than the 5.86 µm detector pixel. The stamp is shifted so that its centre pixel falls on the star position. (The Zemax centre point, 1-based, is row N/2+1, column N/2.) The flux in each detector pixel is the exact overlap-weighted sum of the fine pixels:

    image[y, x] += M_y · (N · PSF_eff) · M_xᵀ

where M is the box-overlap matrix between fine pixels and detector pixels. This works for any ratio of pixel size to PSF sampling.

## 5. Detector noise

The sum of all stars is the noiseless signal S (electrons). For each pixel, `Detector.noisy_image` draws

    e = Poisson( S + N_sky + D(T_ccd) · t ) + Normal(0, σ_read)

and then

    ADU = round( e / gain + bias ),  clipped at 2^12 − 1 = 4095

- D(T) is the dark current (e-/pixel/s) interpolated from `references/gt1930-dark-current.csv`.
- gain = 8 e-/ADU, σ_read = 4.5 e- rms, bias = 0 ADU.
- Poisson sampling of S + N_sky + dark includes the photon shot noise of the stars, of the sky, and the dark-current shot noise.
- The exposure time is limited to 30 s.

## 6. Command line

    python simulated_images.py RA DEC -t EXPOSURE [options] [-o FILE]

Run it from the project directory. It needs network access for the Gaia query, and the packages `numpy`, `scipy`, `astropy`, `astroquery` and `matplotlib` (for `--show`).

| Argument | Meaning | Default |
|---|---|---|
| `RA`, `DEC` | field centre, degrees (ICRS) | required |
| `-t`, `--exposure` | exposure time, s (0 < t ≤ 30) | required |
| `--seeing` | seeing FWHM, arcsec | 1.5 |
| `--temperature` | CCD temperature, °C (sets the dark current) | 0 |
| `--sky-brightness` | sky surface brightness, G_RP mag/arcsec² | 20.5 |
| `--no-sky` | disable the sky background | off |
| `--mag-limit` | ignore stars fainter than this G_RP | none |
| `--diameter` | telescope diameter, m | 1.6 |
| `--obstruction-radius` | central obstruction radius, mm | 200 |
| `--seed` | random seed | none |
| `-o`, `--output` | output FITS file | `simulated.fits` |
| `--show` | display the image when done | off |

The output is a float32 FITS image in ADU. Its header holds CRVAL1/2, EXPTIME, SEEING, CCDTEMP, SKYMAG, SKYE, GAIN, RDNOISE, PIXSIZE and NSTARS.

Example (NGC 188, 10 s, 1.5″ seeing, −5 °C):

    python simulated_images.py 11.798 85.244 -t 10 --seeing 1.5 --temperature -5 -o ngc188-sim.fits

It can also be used from Python:

    from simulated_images import StarFieldSimulator
    sim = StarFieldSimulator(seeing=1.5)
    image = sim.simulate(11.798, 85.244, t_int=10, temperature=-5)  # sky from sim.sky_brightness
    sim.detector.show(image)

## 7. Assumptions and limitations

- Flat f_λ over the G_RP window, no stellar colour dependence. The real G_RP response is not flat, so absolute fluxes carry a colour-dependent error.
- The integral stops at 1.0 µm, the end of the throughput data.
- The sky is uniform and constant: no moon, gradients, extinction or airmass. The 20.5 mag/arcsec² default is an assumption, so set it for your site and conditions.
- No proper motions or epoch propagation (Gaia positions are at epoch 2016.0).
- Seeing is a Gaussian, not a Kolmogorov/Moffat profile, and it is constant over the field.
- The dark current is uniform: no hot pixels, no fixed pattern, no flat-field or vignetting.
- Bias is 0 ADU, so readout noise can produce negative pixel values.
- Stars whose PSF stamp (about ±256 µm) does not overlap the detector are skipped. Stars just outside the edge still add their wings.
- Stars without G_RP photometry are not simulated.
