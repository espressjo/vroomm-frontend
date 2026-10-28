import os
import re

import numpy as np

_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")

_ROW = re.compile(r"Transmission at\s+([-+0-9.eE]+)\s*:\s*([-+0-9.eE]+)")


class _Throughput:
    """Throughput vs wavelength read from a Zemax transmission listing.

    Attributes
    ----------
    wavelength : ndarray
        Wavelength in microns.
    transmission : ndarray
        Fractional transmission.
    """

    default_file = None
    label = "throughput"

    def __init__(self, filename=None):
        filename = filename or self.default_file
        self.filename = filename
        self.wavelength, self.transmission = self._parse(filename)

    @staticmethod
    def _parse(filename):
        wl, tr = [], []
        with open(filename, "r", errors="replace") as f:
            for line in f:
                m = _ROW.search(line)
                if m:
                    wl.append(float(m.group(1)))
                    tr.append(float(m.group(2)))
        if not wl:
            raise ValueError(f"No transmission data found in {filename}")
        wl = np.asarray(wl)
        tr = np.asarray(tr)
        order = np.argsort(wl)
        return wl[order], tr[order]

    def __call__(self, wavelength):
        """Linearly interpolated throughput at `wavelength` (microns).

        Accepts a scalar or array. Values outside the tabulated range
        return 0.
        """
        out = np.interp(wavelength, self.wavelength, self.transmission,
                        left=0.0, right=0.0)
        return float(out) if np.ndim(out) == 0 else out

    def show(self, wavelength=None, ax=None):
        """Plot the throughput curve. Optionally mark `wavelength` points."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavelength, self.transmission, "-", label=self.label)
        if wavelength is not None:
            w = np.atleast_1d(wavelength)
            ax.plot(w, self(w), "o", label="interpolated")
            ax.legend()
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Throughput")
        ax.set_title(f"VROOMM frontend {self.label}")
        ax.grid(alpha=0.3)
        plt.show()
        return ax


class guiding(_Throughput):
    """Throughput of the VROOMM frontend guiding branch."""

    default_file = os.path.join(
        _REF_DIR, "vroomm-frontend-transmission-guiding-arm.txt")
    label = "guiding branch"


class science(_Throughput):
    """Throughput of the VROOMM frontend science branch (fiber)."""

    default_file = os.path.join(_REF_DIR, "vroomm-science-fiber-throughput.txt")
    label = "science branch (fiber)"


if __name__ == "__main__":
    g = guiding()
    print(f"{len(g.wavelength)} points, {g.wavelength[0]}-{g.wavelength[-1]} um")
    print("throughput at 0.75 um:", g(0.75))
    g.show()
