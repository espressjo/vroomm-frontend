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


if __name__ == "__main__":
    psf = ZemaxPSF("./psf/psf-1.txt")
    print(psf)
    print(psf.data.max(), psf.data.argmax())
