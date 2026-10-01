# Misaki — Dust, Hair, and Artefact Removal for Black-and-White Negative Film Scans

Misaki removes dust, hairs and other artefacts from scans of black-and-white negative film.

Scanners pick these up however carefully the negative is brushed. Repairing them in a general-purpose editor tends to leave a smooth patch where the film's grain used to be, and on a grainy frame that patch is what the eye finds. Misaki repairs them while keeping the grain, so the repair disappears rather than merely looking clean.

**Retouching by hand is the way to use it.** Paint over an artefact with **Spot** or **Sweep** and it is repaired inside your stroke: you decide what is dust, and Misaki decides how to fill it. Automatic detection exists but is experimental — see [Global detection](#global-detection-experimental).

## Download

**Windows:** download `Misaki-v1.0.0-windows-x64.zip` from [Releases](https://github.com/Stillll-ph/Misaki/releases), extract the whole folder, and run `Misaki.exe`. No Python is needed. Run it from the extracted folder rather than from inside the zip, since it needs the `_internal` folder beside it.

The executable is not code-signed, so Windows SmartScreen will stop it the first time with "Windows protected your PC". Choose **More info**, then **Run anyway**.

**From source:**

```
pip install -r requirements.txt
python dust_removal_gui.py
```

Developed and tested on Python 3.14 on Windows.

## Using it

1. **Open image...** and choose a scan. Colour scans are converted to greyscale. Bit depth is kept, so a 16-bit TIFF stays 16-bit.
2. Tick **Survey** and step through the frame cell by cell with the arrows above the previews, or `Page Up` and `Page Down`. Every part of the scan comes past once, so nothing is missed because you never panned there.
3. In the **Retouch** tab, paint out what you find: **Spot** for a speck, **Sweep** for a hair or scratch. **Protect** keeps an area from being repaired, and **Eraser** removes a mark.
4. Tick **Show mask** to see in red what will be repaired. **Single pane** shows one large preview and flips between original and result each time you click it.
5. Press **Final render** to process the whole scan, inspect the result, then **Save result...**. Save stays greyed until a render exists, and it writes exactly the pixels you inspected.

Portrait scans are turned on their side while you work and turned back when saved.

## Fills

Each mask can use its own fill, chosen from the **Fill** list.

| Fill | What it does | Grain kept |
| --- | --- | --- |
| **Texture** (default) | Rebuilds the hole from its surroundings, then lays real grain from nearby film over it | 94% |
| **Smooth** | Rebuilds the hole by continuing the surrounding tones across it, without grain | 69% |
| **Median** | Copies values from the film immediately around the speck | Good on small specks; smears once a speck is wider than **Intensity** |
| **Auto** | Median where it can reach, Smooth where it cannot | — |

*Grain kept* compares the fine texture inside a repair with the film that was actually there: 100% would be indistinguishable, 0% a flat patch. The figures come from repairs of real annotated artefacts pasted onto clean film, where the true result is known.

On a pixel-by-pixel accuracy score, Texture comes out about 1.2 dB behind Smooth, because grain that is real film but not *the* grain that was there counts as error. To the eye it is the other way round, which is why Texture is the default.

## Global detection (experimental)

Global detection searches the whole frame for dust by itself. It is in the **Detect** tab and starts switched off.

It is precise but conservative. At its default settings about 90% of what it marks is real dust, but it finds under half of the dust it is shown (precision 0.904, recall 0.450), and 12.4% of real artefacts are too large for it to find at all. Every test it applies is about size and brightness, so it cannot tell a speck from fine bright detail such as foliage, an aerial or a wire.

A detector trained on real corrections is the intended next step, and does not exist yet. **Record my corrections to train a detector** saves your Spot and Protect strokes as training examples, on your own machine only.

Until then, use global detection as a rough first pass on a very dusty frame, and check what it found with **Wide** before saving.

## Known limits

- **Greyscale only.** Colour scans are converted on import and saved as greyscale.
- **Speed.** A 12-megapixel scan takes a few seconds to process, which is why the preview works on part of the frame at a time.
- **Global detection** is experimental, as described above.

## Customising the look

Each empty preview pane can show a background picture. Put your own `pane_left.png` and `pane_right.png` into the `assets` folder — or `Misaki\_internal\assets` in the downloaded build — and they are tinted to match the window. With no pictures there, the panes are plain. The palette is `THEME`, near the top of `dust_removal_gui.py`.

## Notes for developers

**Building the executable:**

```
pyinstaller Misaki.spec --noconfirm --clean
```

**The preview and the saved file.** For speed, the preview processes a padded window rather than the whole scan. The mask it shows is identical to the whole-frame result, and pixels outside the mask never change. Inside the mask, the Smooth and Texture fills can differ between preview and whole frame — by up to about 6% of full scale on one real scan — because the solver behind them works on everything masked at once. This never reaches the saved file: **Final render** processes the whole scan, and **Save** writes exactly that.

The property tests, benchmarks and labelling tools used during development are not included in this repository.

## Acknowledgements

- **[DarkSlide](https://github.com/kilianvivien/DarkSlide)** by Kilian Vivien (MIT) — the idea behind the Texture fill, that grain should be repaired by copying real film rather than generating it. The donor search here is a port of DarkSlide's `findBestPatch`, adapted to greyscale; its copyright notice and licence are in [LICENSE](LICENSE).
- **[FilmDamageSimulator](https://github.com/mcdanieljackson/FilmDamageSimulator)** (MIT) — ten scans of empty 35 mm frames with 12,137 individually annotated artefacts, used as ground truth when measuring the detector and the fills. None of that data is redistributed here.
- **[NumPy](https://numpy.org)**, **[SciPy](https://scipy.org)**, **[scikit-image](https://scikit-image.org)** and **[tifffile](https://github.com/cgohlke/tifffile)** (all BSD-3-Clause) and **[Pillow](https://python-pillow.org)** (MIT-CMU) — the processing and file handling this is built on. scikit-image's `inpaint_biharmonic` is the solver behind the Smooth and Texture fills. The Windows build carries each library's licence in `_internal`.
- **[PyInstaller](https://pyinstaller.org)** (GPL-2.0-or-later, with the bootloader exception that lets a built application keep its own licence) — used to build the executable.
- **Claude** (Anthropic) — pair work on the fills, the benchmarking harness and the property tests.
- The name is an homage to *Welcome to the NHK*.

## Licence

[MIT](LICENSE). The icon in `assets/` is covered by the licence; the licence does not extend to the character depicted in it.
