# VROOMM frontend

Tools for the VROOMM frontend, the optical design of which is done in Zemax.

Currently the repository contains the **guiding image simulator**, in
[`guiding-image-simulator/`](guiding-image-simulator/). It generates noisy
images of a real star field, as seen by the guiding camera (Allied Vision
Prosilica GT1930) on the 1.6 m telescope.

The simulator combines:

- Gaia DR3 star lists (G_RP magnitudes),
- the Zemax PSFs of the frontend, at several field positions,
- the throughput of the guiding branch,
- the camera model (QE, dark current, gain, read noise),
- seeing and sky background.

For the physics and the maths (G_RP magnitude to electrons, seeing, noise),
see [`guiding-image-simulator/simulated-images.md`](guiding-image-simulator/simulated-images.md).

## Installation

Requires Python 3.9 or newer and network access (the Gaia archive is queried
at run time).

```bash
git clone git@github.com:espressjo/vroomm-frontend.git
cd vroomm-frontend/guiding-image-simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The scripts use relative paths for the PSF files, so run them from
`guiding-image-simulator/`.

## Quick start

Simulate a 10 s exposure of NGC 188, with 1.5″ seeing and the CCD at −5 °C:

```bash
cd guiding-image-simulator
python simulated_images.py 11.798 85.244 -t 10 --seeing 1.5 --temperature -5 -o ngc188-sim.fits
```

Add `--show` to display the image. The output is a float32 FITS file in ADU.
The full list of options is in [the simulator documentation](guiding-image-simulator/simulated-images.md#6-command-line)
and with `python simulated_images.py -h`.

| Argument | Meaning | Default |
|---|---|---|
| `RA`, `DEC` | field centre, degrees | required |
| `-t` | exposure time, s (max 30) | required |
| `--seeing` | seeing FWHM, arcsec | 1.5 |
| `--temperature` | CCD temperature, °C | 0 |
| `--sky-brightness` | sky, G_RP mag/arcsec² | 20.5 |
| `-o` | output FITS file | `simulated.fits` |

## Layout

All the code is in `guiding-image-simulator/`.

| File | Contents |
|---|---|
| `simulated_images.py` | Main script and `StarFieldSimulator` class: Gaia download, magnitude to electrons, PSF rendering, noise |
| `psf.py` | `ZemaxPSF` (parser for a Zemax FFT PSF listing) and `PSFSet` (all the PSFs, with lookup by detector location) |
| `throughput.py` | `guiding` and `science` classes: throughput of the guiding branch and of the science fiber branch vs wavelength |
| `detector.py` | `Detector` class: GT1930 camera model and noisy image generation |
| `simulated-images.md` | Physics and maths of the simulation, command-line reference |
| `psf/` | Zemax PSF exports, `psf-0` (on-axis) to `psf-3` |
| `references/` | Throughput tables (Zemax) and the digitized GT1930 QE and dark-current curves |
| `flux-scaling-analysis/` | Photometric zero-point check of the NGC 188 image against Gaia |
| `simulator.py`, `noise.py` | Earlier version of the simulator, superseded by `simulated_images.py` |

## Using the modules from Python

```python
from throughput import guiding
from psf import PSFSet
from detector import Detector
from simulated_images import StarFieldSimulator

# Throughput of the guiding branch, interpolated at any wavelength (µm)
g = guiding()
g(0.75)            # fractional throughput
g.show()           # plot

# PSFs: the right PSF for a detector location (x, y in mm from the centre)
psfs = PSFSet()
psf = psfs.get(0.0, 3.0)    # ZemaxPSF, .data is normalized to peak = 1

# Camera
det = Detector()
print(det)                  # characteristics of the camera
det.show_qe()
det.show_dark_current()
img = det.noisy_image(10.0, temperature=-5)   # dark + read noise only, in ADU
det.show()

# Full star field
sim = StarFieldSimulator(seeing=1.5)
image = sim.simulate(11.798, 85.244, t_int=10, temperature=-5)
```

`Detector.noisy_image` takes an optional `signal` image in electrons, to which
photon noise is added, so it can be used with any source image.

### Adding PSFs

`PSFSet()` loads every `psf/psf-*.txt`. A PSF is chosen according to the
distance to the field centre: along one direction, PSF *i* is used from its
radius to the radius of the next PSF, and the first PSF is used closer in than
the second one. PSFs at different position angles are not assumed to be
equivalent: a location uses the PSFs of the closest direction. See the
`PSFSet` docstring for details.

## Assumptions to keep in mind

- G_RP is assumed to have a flat spectrum over 0.62–1.05 µm. The throughput
  table stops at 1.0 µm.
- The dark current, the sky and the seeing are uniform over the field.
- The GT1930 QE and dark current were digitized from plots; the dark-current
  units (e⁻/pixel/s) should be double-checked.

The complete list is at the end of
[`simulated-images.md`](guiding-image-simulator/simulated-images.md#7-assumptions-and-limitations).

## License

MIT, see [LICENSE](LICENSE).
