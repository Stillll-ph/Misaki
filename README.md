# Misaki — Dust, Hair, and Artefact Removal for Black-and-White Negative Film Scans

Misaki removes dust, hairs and other artefacts from scans of black-and-white negative film. Scanners pick these up however carefully the negative is brushed, and repairing them in a general-purpose editor tends to leave a smooth patch where film grain used to be. This repairs them while leaving the grain intact, which is what makes a repair invisible rather than merely clean.

> **Retouch by hand — that is what this tool is for.**
>
> Paint over an artefact with **Spot** or **Sweep** and it is found and repaired inside your stroke. You decide what is dust; the program decides only how to repair it.
>
> **Global detection is experimental and starts switched off.** Every test it applies is morphological — a size band, a brightness cut, a height above an opening — so it can ask whether something looks like a small bright thing standing above its surroundings, but never whether it is dust or a wire. At its shipping settings it finds under half the dust it is shown (precision 0.904, recall 0.450), 12.4% of real artefacts are too large for it to see, and fine bright structure such as foliage or an aerial can be claimed as dust. **There is no learned model behind it yet** — a U-Net trained on real corrections is the intended next step, and the app's *Record my corrections* option exists to collect the labels for one. Until then, treat it as a rough first pass to check in **Wide**, not as something to save from unattended.

## Running it

```
pip install -r requirements.txt
python dust_removal_gui.py
```

Python 3.9 or later. Tkinter ships with the standard Windows and macOS installers. To build a standalone executable:

```
pyinstaller Misaki.spec --noconfirm --clean
```

## Using it

1. **Open image...** and pick a scan. Colour is converted to greyscale; bit depth is preserved, and portrait scans are turned on their side to work on and turned back on save.
2. Press **Survey** and step through the frame with **◀ ▶** or `Page Up`/`Page Down`. Every cell comes past once, so nothing is missed because you never panned there.
3. Paint out what you find from the **Retouch** tab — **Spot** for a speck, **Sweep** for a hair. Each stroke joins the selected mask; **New mask** starts a fresh one. **Protect** vetoes a region the detector should not touch, and **Eraser** removes whole marks.
4. **Show mask** colours what will be repaired. **Single pane** collapses the two previews into one at full width and flips between original and result on a click, which is the comparison an eye is actually good at.
5. **Final render** processes the whole scan. **Save result...** stays greyed until it has, and then writes exactly the array you inspected — no recomputation between looking and saving.

## How a hole gets filled

| Fill | What it does | Grain kept |
| --- | --- | --- |
| **Texture** (default) | Solves the hole, then copies real grain back over it from matched clean film nearby | 94% |
| **Smooth** | Biharmonic solve — continues the surrounding gradients across the hole | 69% |
| **Median** | Regional median of the unmasked neighbourhood | smears once a speck is wider than *Intensity* |
| **Auto** | Median where it can reach, Smooth where it cannot | |

Measured against known truth on real annotated artefacts, Texture costs about 1.2 dB of PSNR against Smooth and returns 25 points of grain. That trade is the whole design: a smooth patch where dust was is easier to see than grain that is real film but not the grain that was there. PSNR punishes correct grain, so ranking on it alone picks the flat patch every time.

## The guarantee the preview rests on

A padded preview window must give bit-identical pixels to processing the whole frame — otherwise you tune against one thing and save another. Every stage reads a bounded neighbourhood, and `_margin_for` sums their reach so the window is padded past it. Three things hold unconditionally, and the property tests assert each:

- **The mask is exact.** Everything you tune against — the sigmas, Threshold, Spread, and the mask view itself — is identical in a window and in the whole frame.
- **Nothing outside the mask moves**, whichever fill ran.
- **Median is exact**, being a rank filter over a bounded footprint.

The exception is the biharmonic solver behind **Smooth** and **Texture**, and it is wider than a bounded-reach argument suggests. It solves a sparse linear system, and the values it returns for one masked region depend on what else is masked anywhere in the frame. Measured on a real scan: three regions lying entirely inside the padded crop — the furthest 44 px clear of it — differed by up to 4225/65535 between window and frame. Each is perfectly size-independent when it is the only thing masked; adding the frame's other regions back is what moves it, with scikit-image's `split_into_regions` set either way.

So for a solver fill the replacement values inside the mask carry no parity guarantee, and no test of the geometry predicts when they will differ. The differences stay inside the mask and the mask is exact, so what you are judging — where a repair lands, and how much of the frame is claimed — previews truthfully. A save processes the whole frame, so the file on disk is always the correct result; press **Final render** to inspect exactly what will be written.

## Known limits

- **Greyscale only.** Colour scans are converted on import and saved as grey.
- **Global detection is morphological**, with the consequences described above.
- **Speed.** A 12 megapixel scan takes a few seconds a pass, which is why the preview works on a padded window.
- **Large artefacts.** A particle wider than the opening's structuring element is not a bright speck to the detector and is not found.

## Customising the look

Each empty preview pane can carry a background image. Drop `pane_left.png` and `pane_right.png` into `assets/` and they are tinted from the theme and drawn behind the panes; with no files present the panes are simply empty. `THEME` near the top of `dust_removal_gui.py` holds the palette.

## Acknowledgements

- **[DarkSlide](https://github.com/kilianvivien/DarkSlide)** by Kilian Vivien (MIT) — the idea behind the **Texture** fill, that grain should be repaired by copying real film rather than generating it. The donor search here is a port of DarkSlide's `findBestPatch`, adapted to greyscale; its copyright notice and licence are in [LICENSE](LICENSE).
- **[FilmDamageSimulator](https://github.com/mcdanieljackson/FilmDamageSimulator)** (MIT) — ten scans of empty 35 mm frames with 12,137 individually annotated artefacts, used as ground truth when benchmarking the detector and the fills. None of that data is redistributed here.
- **[NumPy](https://numpy.org)** (BSD-3-Clause), **[SciPy](https://scipy.org)** (BSD-3-Clause), **[scikit-image](https://scikit-image.org)** (BSD-3-Clause), **[Pillow](https://python-pillow.org)** (MIT-CMU) and **[tifffile](https://github.com/cgohlke/tifffile)** (BSD-3-Clause) — the processing and I/O this is built on. `inpaint_biharmonic` from scikit-image is the solver behind the Smooth and Texture fills. A frozen build carries each of their licence files in `_internal`.
- **[PyInstaller](https://pyinstaller.org)** (GPL-2.0-or-later, with the bootloader exception that lets a frozen application keep its own licence) — used to build the executable.
- The name is an homage to *Welcome to the NHK*.
- **Claude** (Anthropic) — pair work on the fills, the benchmarking harness and the property tests.

## Licence

[MIT](LICENSE). The icon in `assets/` is covered by the licence; the licence does not extend to the character depicted in it.
