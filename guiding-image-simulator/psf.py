import glob
import os
import re
import numpy as np


class ZemaxPSF:
    """Parser for a Zemax 'Listing of FFT PSF Data' text file.

    Attributes
    ----------
    data : np.ndarray
        PSF image (normalized to peak = 1.0).
    shape : tuple
        Shape of `data` (rows, cols), taken from the parsed array.
    reference_coordinate : tuple of float
        Reference coordinates on the detector (x, y), in lens units (mm).
    scale : float
        Data spacing (µm per pixel).
    field : tuple of float
        Field position (x, y) in degrees.
    """

    _FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"

    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.shape = None
        self.reference_coordinate = None
        self.scale = None
        self.field = None
        self._parse()

    # ------------------------------------------------------------------
    @staticmethod
    def _read_text(filename):
        """Zemax listings are often UTF-16; fall back to UTF-8 / latin-1."""
        with open(filename, "rb") as f:
            raw = f.read()
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        for enc in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeError:
                continue
        raise ValueError("Cannot decode file")

    @staticmethod
    def _to_floats(line):
        """Return list of floats if the whole line is numeric, else None."""
        tokens = line.split()
        if len(tokens) < 2:
            return None
        try:
            return [float(t) for t in tokens]
        except ValueError:
            return None

    def _parse(self):
        text = self._read_text(self.filename)
        F = self._FLOAT
        rows = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            # --- numeric data rows -------------------------------------
            values = self._to_floats(line)
            if values is not None:
                rows.append(values)
                continue

            # --- header information ------------------------------------
            # e.g. "0.5500 µm at 0.0000, 0.0450 (deg)."
            m = re.search(rf"at\s+({F})\s*,\s*({F})\s*\(deg\)", line)
            if m:
                self.field = (float(m.group(1)), float(m.group(2)))
                continue

            # e.g. "Data spacing is 0.500 µm."
            m = re.search(rf"Data spacing is\s+({F})", line)
            if m:
                self.scale = float(m.group(1))
                continue

            # e.g. "Reference Coordinates: 0.00000E+00, 5.57323E+00"
            m = re.search(rf"Reference Coordinates:\s*({F})\s*,\s*({F})", line)
            if m:
                self.reference_coordinate = (float(m.group(1)), float(m.group(2)))
                continue

        if not rows:
            raise ValueError("No PSF data found in file")

        lengths = {len(r) for r in rows}
        if len(lengths) != 1:
            raise ValueError(f"Ragged data rows, lengths found: {sorted(lengths)}")

        self.data = np.array(rows, dtype=float)
        self.shape = self.data.shape

        missing = [n for n in ("field", "scale", "reference_coordinate")
                   if getattr(self, n) is None]
        if missing:
            raise ValueError(f"Missing header information: {missing}")
    def show(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval, ImageNormalize


        # Apply ZScale
        norm = ImageNormalize(self.data, interval=ZScaleInterval())

        # Display
        plt.imshow(self.data, origin="lower", cmap="gray", norm=norm)
        plt.colorbar()
        plt.show()
    def __repr__(self):
        return (f"ZemaxPSF(shape={self.shape}, field={self.field} deg, "
                f"scale={self.scale} um, ref={self.reference_coordinate})")


class PSFSet:
    """All the PSFs of the instrument, with lookup by detector location.

    Selection follows the radial-zone rule of `GuidingSimulator.psf_index`:
    along one direction, PSF i is used for r in [r_i, r_{i+1}) and the last
    PSF beyond it, so a location closer than the second PSF uses the first.

    PSFs are not assumed to be symmetric about the field centre. Each PSF has
    a position angle (from its reference coordinate); a query is first matched
    to the closest direction, then the radial rule is applied among the PSFs
    of that direction. PSFs on the axis (r = 0) are shared by every direction.
    With PSFs along a single direction only (current data), every location
    uses that direction.

    Parameters
    ----------
    files : list of str, optional
        Zemax PSF files (any order). Defaults to ``psf/psf-*.txt`` next to
        this script.
    angle_tol_deg : float
        PSFs whose position angles differ by less than this are considered to
        lie along the same direction.
    """

    def __init__(self, files=None, angle_tol_deg=1.0):
        if files is None:
            here = os.path.dirname(os.path.abspath(__file__))
            files = sorted(glob.glob(os.path.join(here, "psf", "psf-*.txt")))
        if not files:
            raise ValueError("No PSF files given")
        self.psfs = [ZemaxPSF(f) for f in files]

        xy = np.array([p.reference_coordinate for p in self.psfs])
        self.radius_mm = np.hypot(xy[:, 0], xy[:, 1])
        self.angle = np.degrees(np.arctan2(xy[:, 1], xy[:, 0]))

        on_axis = self.radius_mm < 1e-9
        self._axis = list(np.flatnonzero(on_axis))

        # Group off-axis PSFs by direction: list of (angle, [indices by radius])
        self._dirs = []
        for i in np.flatnonzero(~on_axis):
            for d in self._dirs:
                if abs(_wrap(self.angle[i] - d[0])) < angle_tol_deg:
                    d[1].append(i)
                    break
            else:
                self._dirs.append((self.angle[i], [i]))
        for _, idx in self._dirs:
            idx.sort(key=lambda i: self.radius_mm[i])

    # ------------------------------------------------------------------
    def index(self, x_mm, y_mm):
        """Index (into `self.psfs`) of the PSF for a detector location (mm)."""
        r = np.hypot(x_mm, y_mm)
        if self._dirs:
            theta = np.degrees(np.arctan2(y_mm, x_mm)) if r > 0 else 0.0
            _, chain = min(self._dirs,
                           key=lambda d: abs(_wrap(theta - d[0])))
        else:
            chain = []
        chain = self._axis + chain
        radii = self.radius_mm[chain]
        k = np.searchsorted(radii, r, side="right") - 1
        return chain[int(np.clip(k, 0, len(chain) - 1))]

    def get(self, x_mm, y_mm):
        """PSF (ZemaxPSF) for a detector location (x, y) in mm from the centre."""
        return self.psfs[self.index(x_mm, y_mm)]

    __call__ = get

    def __len__(self):
        return len(self.psfs)

    def __iter__(self):
        return iter(self.psfs)

    def __getitem__(self, i):
        return self.psfs[i]

    def __repr__(self):
        return f"PSFSet({len(self)} PSFs, {len(self._dirs)} direction(s))"


def _wrap(a):
    """Wrap an angle difference (deg) to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


if __name__ == "__main__":
    S = PSFSet()
    p1 = S.get(1,1)
    p2 = S.get(2,2)
    p3 = S.get(3,3)
    p4 = S.get(4,4)
    p5 = S.get(5,5)
    p1.show()
    p2.show()
    p3.show()
    p4.show()
    p5.show()
    #manage single PSF
    """
    psf = ZemaxPSF("./psf/psf-3.txt")
    print(psf)
    print(psf.data.max(), psf.data.argmax())
    psf.show()
    """
