"""Allied Vision Prosilica GT1930 (Sony IMX174) guiding-camera model.

Generates noisy images in ADU from a noiseless electron image (e.g. the PSFs
rendered by the simulator). QE and dark current come from the digitized plots
in references/.
"""
import os

import numpy as np

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")


def _load_curve(filename):
    """Read a 2-column CSV (with header), return x-sorted unique (x, y)."""
    data = np.loadtxt(os.path.join(_REF_DIR, filename), delimiter=",",
                      skiprows=1)
    x, i = np.unique(data[:, 0], return_index=True)
    return x, data[i, 1]


class Detector:
    """Allied Vision GT1930 camera.

    Parameters
    ----------
    gain : float
        Electrons per ADU.
    read_noise : float
        Readout noise, electrons rms.
    max_exposure : float
        Longest allowed exposure, seconds.
    """

    name = "Allied Vision Prosilica GT1930"
    sensor = "Sony IMX174 (CMOS, global shutter)"
    shape = (1216, 1936)          # (ny, nx) pixels
    pixel_size = 5.86             # µm
    bit_depth = 12

    def __init__(self, gain=8.0, read_noise=4.5, max_exposure=30.0):
        self.gain = gain
        self.read_noise = read_noise
        self.max_exposure = max_exposure
        self.qe_wavelength, self.qe_values = _load_curve("gt1930-qe.csv")
        self.dark_temperature, self.dark_values = _load_curve(
            "gt1930-dark-current.csv")
        self.image = None         # last image from `noisy_image`

    @property
    def max_adu(self):
        return 2 ** self.bit_depth - 1

    # ------------------------------------------------------------------
    def qe(self, wavelength):
        """Quantum efficiency at `wavelength` (µm); 0 outside the tabulated range."""
        out = np.interp(wavelength, self.qe_wavelength, self.qe_values,
                        left=0.0, right=0.0)
        return float(out) if np.ndim(out) == 0 else out

    def dark_current(self, temperature):
        """Dark current (e-/pixel/s) at CCD `temperature` (°C).

        Held constant beyond the tabulated temperature range.
        """
        out = np.interp(temperature, self.dark_temperature, self.dark_values)
        return float(out) if np.ndim(out) == 0 else out

    # ------------------------------------------------------------------
    def noisy_image(self, t_int, temperature=0.0, signal=None, sky=0.0,
                    bias=0.0, rng=None):
        """Generate a noisy image in ADU (stored in `self.image` and returned).

        Parameters
        ----------
        t_int : float
            Exposure time in seconds (0 < t_int <= max_exposure).
        temperature : float
            CCD temperature in °C, sets the dark current.
        signal : np.ndarray, optional
            Noiseless source image in electrons over the exposure (e.g. PSFs
            superimposed), shape `self.shape`. Photon noise is added to it.
        sky : float
            Mean sky background, electrons per pixel over the exposure.
        bias : float
            Bias offset in ADU (default 0, so negative values from readout
            noise are kept rather than clipped).
        rng : int or np.random.Generator, optional
            Seed / generator.

        Noise terms: Poisson(signal + sky + dark * t_int) and Gaussian readout
        noise, in electrons, converted to ADU with the gain, rounded to
        integers and clipped at saturation.
        """
        if not 0 < t_int <= self.max_exposure:
            raise ValueError(f"t_int must be in (0, {self.max_exposure}] s")
        rng = np.random.default_rng(rng)

        mean = np.full(self.shape, sky + self.dark_current(temperature) * t_int)
        if signal is not None:
            signal = np.asarray(signal, float)
            if signal.shape != self.shape:
                raise ValueError(f"signal shape {signal.shape} != {self.shape}")
            mean = mean + np.clip(signal, 0, None)
        electrons = rng.poisson(mean) + rng.normal(0.0, self.read_noise,
                                                   self.shape)
        adu = np.round(electrons / self.gain + bias)
        self.image = np.minimum(adu, self.max_adu)
        return self.image

    # ------------------------------------------------------------------
    def show_qe(self):
        """Plot the quantum efficiency vs wavelength."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.qe_wavelength, self.qe_values * 100)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Quantum efficiency (%)")
        ax.set_title("GT1930 quantum efficiency")
        ax.grid(alpha=0.3)
        plt.show()
        return ax

    def show_dark_current(self):
        """Plot the dark current vs CCD temperature."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(self.dark_temperature, self.dark_values)
        ax.set_xlabel("CCD temperature (°C)")
        ax.set_ylabel("Dark current (e-/pixel/s)")
        ax.set_title("GT1930 dark current")
        ax.grid(alpha=0.3)
        plt.show()
        return ax

    def show(self, image=None):
        """Display an image (default: the last one generated), ZScale stretch."""
        import matplotlib.pyplot as plt
        from astropy.visualization import ImageNormalize, ZScaleInterval

        image = self.image if image is None else image
        if image is None:
            raise ValueError("No image yet: call noisy_image() first")
        norm = ImageNormalize(image, interval=ZScaleInterval())
        fig, ax = plt.subplots()
        im = ax.imshow(image, origin="lower", cmap="gray", norm=norm)
        fig.colorbar(im, label="ADU")
        ax.set_title("GT1930 simulated image")
        plt.show()
        return ax

    # ------------------------------------------------------------------
    def __str__(self):
        ny, nx = self.shape
        qe_pk = self.qe_values.argmax()
        return "\n".join([
            self.name,
            f"  Sensor           : {self.sensor}",
            f"  Array size       : {nx} x {ny} pixels",
            f"  Pixel size       : {self.pixel_size} x {self.pixel_size} um",
            f"  Sensor size      : {nx * self.pixel_size / 1e3:.2f} x "
            f"{ny * self.pixel_size / 1e3:.2f} mm",
            f"  Bit depth        : {self.bit_depth} bit (max {self.max_adu} ADU)",
            f"  Gain             : {self.gain} e-/ADU",
            f"  Read noise       : {self.read_noise} e- rms",
            f"  Max exposure     : {self.max_exposure} s",
            f"  QE               : peak {self.qe_values[qe_pk] * 100:.1f}% at "
            f"{self.qe_wavelength[qe_pk]:.3f} um "
            f"({self.qe_wavelength[0]:.2f}-{self.qe_wavelength[-1]:.2f} um)",
            f"  Dark current     : {self.dark_values.min():.2f}-"
            f"{self.dark_values.max():.2f} e-/pix/s over "
            f"{self.dark_temperature[0]:.0f} to {self.dark_temperature[-1]:.0f} C",
        ])


if __name__ == "__main__":
    det = Detector()
    print(det)
    det.show_qe()
    det.show_dark_current()
    det.noisy_image(10.0, temperature=0.0)
    det.show()
