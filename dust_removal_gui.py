"""
Misaki - dust, hair, and artefact removal for black-and-white negative film
scans.

Dust is segmented with a difference of Gaussians combined with a brightness
threshold and a test that the speck stands proud of its surroundings, then
repaired by one of four fills so the rest of the image keeps its sharpness and
its grain. See README.md.

This works on grayscale: colour scans are converted to gray on import and
saved as gray.

Run with:  python dust_removal_gui.py
"""

import csv
import json
import math
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import skimage.color
import skimage.filters
import skimage.filters.rank
import skimage.io as skio
import tifffile
from skimage.morphology import disk, white_tophat
from PIL import Image, ImageTk

# Explicit rather than attribute access, so PyInstaller sees the dependency
# through skimage's lazy module loader.
from skimage.filters.rank import median as rank_median
from skimage.filters import difference_of_gaussians, gaussian
from skimage.restoration import inpaint_biharmonic
from scipy.ndimage import (binary_fill_holes, convolve, distance_transform_edt,
                           find_objects, label, uniform_filter)

APP_TITLE = "Misaki"
# Bumped per release. Shown in the footer while no scan is open, which is the
# one place it can sit without competing with the file information that
# replaces it - and it is what identifies a copied exe once releases exist.
APP_VERSION = "1.0.0"

DEFAULTS = {
    "window": 256,
    # Survey mode only: how big a section is. Replaced by `survey_size` for the
    # frame in hand the moment the mode is entered, so this is only what the
    # slider is built with. Like `window` it never reaches the result.
    "division": 512,
    "intensity": 6,
    "sig_min": 2.0,
    "sig_max": 4.0,
    "threshold": 47.1,      # percent of full scale, i.e. 120/255 in 8-bit terms
    "spread": 0.5,
}

# Slider values that decide what you are *shown* rather than what is computed.
# They never reach `remove_dust`, never enter the render signature, and never
# invalidate a render - so anything treating DEFAULTS as the algorithm's
# arguments has to take them back out again.
VIEW_ONLY = ("window", "division")

BRUSH_DEFAULTS = {
    "size": 10,
    # A mark's own feather, starting from the global default.
    "mark_spread": 0.5,             # brush radius, in image pixels
    # Percent; where the ridge cut sits inside a stroke. Measured against real
    # Tri-X grain: 35% recovers about two thirds of a hair only 5/255 above
    # its background while touching 8% of the grain in the stroke. Unlike the
    # score this replaced, it keeps paying off as it is raised.
    "sensitivity": 35.0,
}

# The brush hunts hairs, which are one or two pixels wide, so it uses its own
# much finer band than the global pass. Sigma 2.0 - the default sig_min -
# blurs a hair away before the difference of Gaussians ever sees it, which is
# why the brush used to plateau however far sensitivity was raised.
BRUSH_SIGMA = (0.6, 1.6)

# Sensitivity maps onto an absolute cut on the brush ridge score. Both ends
# are read off a threshold sweep over real grain: the top barely touches it,
# the bottom is a blunt instrument that takes a third of it.
BRUSH_TOP = 3.0e-4
BRUSH_FLOOR = 3.3e-5

# How the hole left by a detected particle gets filled.
#
# Texture is the default: the solver's reconstruction with real grain copied
# back over it. Its structure is Smooth's, pixel for pixel - the same solve -
# so everything below about why the solver beats the median still decides what
# the default is built on. What Texture adds is grain, 73% of the surrounding
# film against Smooth's 61%, and what it costs is 0.28 dB of PSNR and about
# 1.6x the time. That gap is the metric penalising the thing being asked for:
# transplanted grain has the right statistics and the wrong realisation, so it
# is necessarily further from the original per pixel than a flat patch is. The
# judgement is which is easier to see on a finished frame - a smooth spot where
# dust was, or grain that is real film but not the grain that was there - and
# the dropdown is one click if you disagree with it on a given scan.
#
# Auto splits the mask between the median and the solver
# on the assumption that much of it is not dust, where the median's grain is
# worth keeping - which held while the detector masked 4.3% of a frame, most of
# it false positives. Requiring a detection to stand proud of its surroundings
# cut that to about 1.2%, so what is left is mostly real dust, and on real dust
# the solver is far ahead: measured against pasted dust with a known truth,
# error of 21.9 for Smooth against 29.5 for Auto and 73.8 for Median. Median
# still wins on the false positives that remain, but that error is a slightly
# different rendition of film, while a median smear across real dust is plain
# to see.
# Whether the global search runs at all when a scan is opened. Off: measured
# over the 18 pasted-dust plates the automatic pass finds most of the dust, but
# on a real frame it also masks about 1.2% of the picture, and what that 1.2%
# lands on is decided by four sliders nobody has touched yet. Starting it off
# means the first thing an opened scan shows is the scan, and every repair in
# the frame is one that was asked for. Turn it on from the Detect tab once the
# frame is worth the sweep.
DEFAULT_AUTO = False

FILL_AUTO = "auto"
FILL_MEDIAN = "median"
FILL_BIHARMONIC = "biharmonic"
FILL_TEXTURE = "texture"
DEFAULT_FILL = FILL_TEXTURE

# Texture transplant, after the approach in DarkSlide (MIT, Kilian Vivien):
# reconstruct the hole's *structure* with the solver as Smooth does, then add
# back a mean-subtracted residual copied from clean film nearby, so the repair
# carries real grain rather than none.
#
# The donor search in `_transplant_grain` is a port of that project's
# `findBestPatch` - the three search radii, the sixteen angles and the shape of
# the score are theirs, adapted from RGB to greyscale. Their copyright notice
# and licence are in LICENSE, under THIRD-PARTY NOTICE.
#
# The point of copying rather than synthesising is that nothing has to be
# estimated. An earlier attempt here generated grain from an amplitude measured
# near the repair, and repairs cluster where there is structure - exactly where
# no clean film is in range - so the estimate ranged from a fifth to six times
# the truth. A residual lifted off real film is whatever amplitude that film
# had, with no parameter to get wrong.
#
# Both radii are capped rather than scaled with the blob, because a preview
# window is padded by a fixed margin: a donor search that reached further as the
# mask grew would read outside what the window can see, and the preview would
# stop matching the save.
TEXTURE_PATCH = 12          # radius of the donor patch, px
TEXTURE_REACH = 32          # furthest a donor may sit from the hole, px
# How much matching *grain energy* counts against matching tone when choosing a
# donor. DarkSlide weights the equivalent term at about 0.16 once its gradient
# is converted to the same units as its colour difference; grain is the whole
# point of this fill, so it starts higher here and is swept in bench.py.
TEXTURE_GRAIN_WEIGHT = 8.0
# What counts as grain rather than picture. Everything broader than this is
# removed from the donor before it is copied, so only the fine texture travels.
TEXTURE_GRAIN_SIGMA = 1.5

# One line per method, shown for whichever is selected. Describing all three at
# once meant two thirds of the sentence was about something you had not chosen.
FILL_NOTES = {
    FILL_BIHARMONIC: "Rebuilds the film across the speck.",
    FILL_MEDIAN: "Keeps grain, but smears wide specks.",
    FILL_AUTO: "Median on small specks, Smooth on wide.",
    # Two words shorter than it reads naturally, because this is the default
    # and so the one that has to fit behind NEW_MASK_PREFIX: at 283 px against
    # the 278 a hint has, the Retouch panel opened with a two-line note.
    FILL_TEXTURE: "Rebuilds, then copies nearby grain back.",
}

# What the Retouch fill box says when no mask is selected, where it is setting
# the method the next mask is born with rather than describing an existing one.
# Kept this short deliberately: with the longest note behind it the whole line
# measures 262 px against the 278 a hint has, so it never wraps to two.
NEW_MASK_PREFIX = "New masks: "

# The DoG response is thresholded at this fixed value.
# difference_of_gaussians normalises to 0..1 whatever the input depth, so this
# one number is valid for both 8- and 16-bit images.
DOG_LEVEL = 0.05

# A difference of Gaussians is positive on the bright side of a *dark* edge -
# the wide blur is dragged down by the dark side while the narrow one is not -
# so the sky beside a dark wire passes both the DoG and the brightness test and
# gets masked as dust. Requiring a real local peak on top of that fixes it: the
# height of the pixel above an opening of the image, which is unmoved by a
# neighbouring dark edge because a min-max filter does not overshoot.
#
# An opening only deletes features that fit inside its structuring element, so
# TOPHAT_RADIUS is also the largest speck the detector will accept - a particle
# bigger than this survives the opening, reads as height zero and is thrown
# away. Enlarging it is not free either: closely spaced dark structures, like
# the elements of a TV antenna, leave strips of sky narrower than a big element,
# and those strips then read as bright features. Measured on the antenna frame
# against dust pasted onto a different negative's film:
#
#     radius    antenna outline masked    largest speck still found
#       none            48.8%                    any
#        8              13.3%                    ~7 px
#       12              20.3%                    ~11 px
#       16              28.8%                    ~15 px
#
# 12 keeps most of the benefit and covers dust up to about 11 px in radius.
# Raise it if specks are being left behind; the Brush repairs any it misses.
TOPHAT_RADIUS = 12
TOPHAT_LEVEL = 0.14
# The octagonal decomposition is 2.5x faster than a plain disc and differs on
# under 0.1% of pixels.
TOPHAT_DISK = disk(TOPHAT_RADIUS, decomposition="sequence")

# A repair keeps only about 42% of the grain around it, so a finished frame
# carries slightly flat patches where dust was. Synthesising grain to fill them
# was tried and removed: the amplitude has to be estimated from clean film near
# the repair, and repairs cluster where there is structure, exactly where no
# clean film is in range. The estimate came out anywhere from a fifth to six
# times the true grain depending on where it landed - grit around every
# high-contrast edge - and suppressing that made it do nothing at all. It needs
# an estimator that widens its search when local support is thin, rather than a
# threshold.

# The rank median histograms every intensity level, so a full 16-bit filter is
# roughly 75x slower than 8-bit. Binning the *replacement values* to 12 bits
# brings a 12MP scan back to a few seconds. This affects only the pixels being
# repaired; everything outside the dust mask keeps full precision.
MEDIAN_BITS = 12

OPEN_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
    ("JPEG", "*.jpg *.jpeg"),
    ("PNG", "*.png"),
    ("TIFF", "*.tif *.tiff"),
    ("All files", "*.*"),
]
QUEUE_EXTS = tuple(p.lstrip("*") for p in OPEN_TYPES[0][1].split())


def queue_from_folder(folder):
    """
    The scans in a folder, in name order, to work through one after another.

    This app's own outputs are left out. A folder that has been half worked
    through holds a `_dustfree` file beside each finished scan, and queueing
    those would put every finished frame back in front of you as if it were
    new - and saving it again would write `_dustfree_dustfree`.
    """
    out = []
    for name in sorted(os.listdir(folder), key=str.lower):
        stem, ext = os.path.splitext(name)
        if ext.lower() in QUEUE_EXTS and not stem.lower().endswith("_dustfree"):
            out.append(os.path.join(folder, name))
    return out


# Keyboard shortcuts, in the order the settings window lists them: an internal
# name, what it is called on screen, and the key it ships with. The list is the
# single source for the defaults, for what can be rebound, and for what the
# window shows - adding a shortcut anywhere else would give a key nobody can
# find or change.
KEY_ACTIONS = (
    ("open", "Open image", "<Control-o>"),
    ("open_folder", "Open folder", "<Control-Shift-O>"),
    ("queue_prev", "Previous scan", "<Control-Left>"),
    ("queue_next", "Next scan", "<Control-Right>"),
    ("queue_skip", "Skip scan", "<Control-Shift-Right>"),
    ("render", "Final render", "<Control-r>"),
    ("save", "Save result", "<Control-s>"),
    ("delete_mark", "Delete selected mask", "<Control-z>"),
    ("new_mark", "New mask", "<Control-n>"),
    ("brush_down", "Smaller brush", "<bracketleft>"),
    ("brush_up", "Larger brush", "<bracketright>"),
    ("prev_section", "Previous cell", "<Prior>"),
    ("next_section", "Next cell", "<Next>"),
    ("toggle_survey", "Survey mode on/off", "<F2>"),
    ("toggle_mask", "Show mask", "<F3>"),
    ("toggle_single", "Single pane", "<F4>"),
)
KEYBIND_FILE = "keybinds.json"

# What a keysym is called on screen. Anything not listed is shown as Tk names
# it, which is already readable for letters, digits and the function keys.
KEY_NAMES = {
    "Prior": "Page Up", "Next": "Page Down", "bracketleft": "[",
    "bracketright": "]", "space": "Space", "Return": "Enter",
    "comma": ",", "period": ".", "minus": "-", "equal": "=",
    "slash": "/", "backslash": "\\", "semicolon": ";", "quoteright": "'",
    "grave": "`", "Escape": "Esc", "Delete": "Del", "BackSpace": "Backspace",
}
# Keys the window will not take, because taking them would break the way the
# rest of the interface is driven.
KEY_REFUSED = {"Tab", "ISO_Left_Tab", "Escape", "Return", "space"}


def key_label(sequence):
    """'<Control-o>' as something to print: 'Ctrl + O'."""
    if not sequence:
        return "unset"
    parts = sequence.strip("<>").split("-")
    out = []
    for part in parts[:-1]:
        out.append({"Control": "Ctrl", "Mod1": "Alt"}.get(part, part))
    last = parts[-1]
    out.append(KEY_NAMES.get(last, last.upper() if len(last) == 1 else last))
    return " + ".join(out)


def key_from_event(event):
    """The binding a keypress should become, or None if it cannot be one."""
    sym = event.keysym
    if sym in KEY_REFUSED or sym.startswith(("Control_", "Shift_", "Alt_",
                                             "Meta_", "Super_", "Win_")):
        return None
    parts = []
    if event.state & 0x0004:
        parts.append("Control")
    if event.state & 0x20000:
        parts.append("Alt")
    # A shifted letter already arrives as its uppercase keysym, so naming the
    # modifier as well gives Tk a sequence no keypress can ever match.
    if event.state & 0x0001 and len(sym) > 1:
        parts.append("Shift")
    return "<%s>" % "-".join(parts + [sym])


def load_keys():
    """Shipped shortcuts, with any the user has changed laid over them."""
    keys = {name: default for name, _label, default in KEY_ACTIONS}
    try:
        with open(data_path(KEYBIND_FILE), encoding="utf-8") as handle:
            saved = json.load(handle)
    except Exception:
        return keys                     # missing or unreadable: ship defaults
    for name in keys:
        value = saved.get(name)
        if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
            keys[name] = value
        elif value is None and name in saved:
            keys[name] = ""             # deliberately unbound
    return keys


def save_keys(keys):
    """Write the shortcuts back, and say whether it worked."""
    try:
        with open(data_path(KEYBIND_FILE), "w", encoding="utf-8") as handle:
            json.dump(keys, handle, indent=2, sort_keys=True)
        return True
    except Exception:
        return False


def _rgb(value):
    """'#rrggbb' to a plain (r, g, b) tuple."""
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def data_path(*parts):
    """
    Somewhere writable, unlike `resource_path`.

    A frozen build unpacks its assets into a temporary directory that is
    deleted on exit, so anything the app *produces* has to go next to the
    executable instead of next to its resources.
    """
    base = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def resource_path(*parts):
    """Locate a bundled asset, both from source and inside a frozen build."""
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


# The plum of her uniform and the dusty mauve of the light behind her, sampled
# from the artwork the app is named after and then checked rather than trusted.
#
# The palette this replaced was the same hues at their first-guess values, and
# three of its pairings did not hold up: hints sat at 3.44:1 where 8pt text
# needs 4.5, white on the accent at 4.29:1, and the slider trough was 1.03:1
# against the panel, which is to say invisible. Every pairing here was measured.
# Body text 12.1:1, hints 5.6:1, white on the accent 5.6:1, no failures.
#
# Two things carried over from working through this properly:
#
# The accent is split by job. A *fill* only has to be seen; a thin *stroke* - a
# focus ring, a selected edge - carries meaning on its own and has to clear
# 3:1. `accent` fills, `accent_dark` draws lines, at 8.0:1.
#
# `canvas` sits behind the previews and is deliberately close to neutral: its
# channels span 5 levels where the old one spanned 14. The scans are greyscale,
# and a strongly tinted surround makes neutral film read as its complement by
# simultaneous contrast - which is the one judgement this window exists to
# support.
#
# The window itself is lighter than it used to be, 0.894 against 0.801, which
# is where the airiness comes from. The plum is the only saturated thing in it.
THEME = {
    "bg": "#f7f1f7",        # window and panels
    "raised": "#fdfafd",    # controls sitting on it
    "canvas": "#ded9dd",    # behind the previews; near-neutral on purpose
    "edge": "#c9b4c9",      # borders
    "ink": "#3c283c",       # text, the plum of her uniform rather than black
    "faint": "#6f5a6f",     # hints
    "accent": "#7d5a93",    # fills; dark enough to carry white text
    "accent_dark": "#5a3f6c",   # strokes, and the pressed state
    "accent_soft": "#f0e8f5",   # troughs and unselected tabs
    "dim": "#b0a0b0",
    # The preview panes' own border, darker than `edge` because it is the one
    # line in the window that says where the picture stops. At `edge` it
    # measured 1.74:1 against the panel - not a weak boundary, an invisible
    # one. This is the lightest plum on the same hue that clears the 3:1 a
    # non-text element carrying meaning needs: 3.20:1 against the window.
    "pane_edge": "#9d7e9d",       # a control nothing is reading
}
# Tahoma is the period-correct Windows UI face; Trebuchet is rounder and reads
# better at small sizes. Both ship with Windows, so neither needs bundling.
FONT = ("Trebuchet MS", 9)
FONT_SMALL = ("Trebuchet MS", 8)
# The mask list carries columns now, and Trebuchet is proportional, so the
# figures would not line up under each other without this.
FONT_MONO = ("Consolas", 9)
FONT_BOLD = ("Trebuchet MS", 9, "bold")
FONT_TAB = ("Trebuchet MS", 11, "bold")     # the selected tab only

# The navigator used to take every pixel the controls left over, which on a
# tall window meant a huge thumbnail above cramped sliders. It is fixed now,
# and the sliders get the slack instead. A portrait 35mm frame is 2:3, so this
# height still leaves the thumbnail about 125 px wide - small, but the point of
# it is where you are in the frame, not what is in the frame.
# Every scan is landscape by the time it reaches the sidebar - portrait ones
# are turned on import - so the thumbnail can be sized to fill the column
# rather than leaving a quarter of it grey. 200 is SIDEBAR_WIDTH over 3:2,
# the shape a 35mm frame arrives in.
NAV_HEIGHT = 200

# The overlap bands on the Cells thumbnail. Named and separate from the
# magenta that marks *where you are*, because the two want different weights:
# where-you-are has to be found at a glance, the seams only have to be
# available when looked for. Swept against a real thumbnail at five strengths:
# the magenta the window rectangle uses read as a grille laid over the picture,
# and two steps lighter than this it survives only on the dark half of the frame,
# which is worse than not drawing it. This one is legible on both.
SEAM_TINT = "#e6b4d9"

# The margin around the previews: window edge to the left pane, and right
# pane to the sidebar. One number so the two cannot drift apart.
SIDE_GAP = 12

# The sidebar column, and how wide a hint may run inside it before wrapping -
# the width less the tab padding and the scrollbar when it shows.
SIDEBAR_WIDTH = 300
HINT_WRAP = SIDEBAR_WIDTH - 22

# How much context to keep around a recorded correction. Big enough that a
# detector sees the speck in its surroundings rather than in isolation, small
# enough that a working session does not fill a disk.
LABEL_PATCH = 128

# The empty state's backdrop. 0.22 is picked off a strength sweep rendered over
# the panel colour: below it the art is barely there, above it the picture
# starts competing with the controls instead of sitting under them.
BACKDROP_STRENGTH = 0.22
# One picture per pane, left then right. Both are reduced to luminance and
# tinted from THEME here, so the palette stays in one place; the files
# themselves carry only a normalised gray and a feathered alpha border.
BACKDROP_FILES = ("pane_left.png", "pane_right.png")

# The gap above each slider row. Fixed rather than shared out from whatever
# the panel has spare: expanding spacers spread the two Retouch sliders across
# the whole tab, which read as unrelated controls rather than a pair.
SLIDER_GAP = 14

# How near a spot's rim, in screen pixels, counts as grabbing it to resize.
SPOT_GRAB = 7

# The old cap of 512 was smaller than any screen, so Wide could only ever
# magnify a small crop. Timed on a 12MP 16-bit scan, a window this
# size is 0.38s on Smooth and 0.62s on Auto - still quick enough to drag a
# slider against - where the whole frame is 3.45s.
WINDOW_MAX = 1024


# --------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------

def _blur(img, scale):
    """Gaussian blur at the given sigma."""
    return gaussian(img, sigma=scale)


def _expand(mask, scale):
    """Grow a binary mask into a soft-edged blending weight."""
    blurred = (10 * _blur(mask, scale)) > 0.5
    expanded = (blurred + mask) > 0
    return _blur(expanded, scale / 1.5)


def _overlay(img, img2, mask):
    """Blend img over img2 using mask as a per-pixel weight."""
    return img * mask + img2 * (1 - mask)


def full_scale(dtype):
    """Largest value the working dtype holds."""
    return 65535 if np.dtype(dtype) == np.uint16 else 255


def brush_level(sensitivity):
    """Ridge-score cut used inside brushed regions, from a 0-100 sensitivity."""
    return BRUSH_TOP * (BRUSH_FLOOR ** (sensitivity / 100.0))


# Second-derivative kernels, written out rather than reached for through a
# library so the whole score stays three convolutions wide.
_K_XX = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float64)
_K_YY = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float64)
_K_XY = 0.25 * np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], dtype=np.float64)


def _line_likeness(values):
    """
    How much each pixel looks like part of a thin line rather than a blob.

    A ridge makes the Hessian's two eigenvalues very different - one large
    across the line, one near zero along it - while isotropic noise like film
    grain makes them similar. The gap between them is therefore what separates
    a hair from the grain it lies on.

    This is the cheap single-scale form of a Frangi filter. Measured against
    skimage's multi-scale `frangi` it matched or beat it on faint hairs at a
    fifteenth of the cost.
    """
    ixx = convolve(values, _K_XX, mode="nearest")
    iyy = convolve(values, _K_YY, mode="nearest")
    ixy = convolve(values, _K_XY, mode="nearest")
    trace = ixx + iyy
    root = np.sqrt(np.maximum(0.0, trace * trace / 4.0
                              - (ixx * iyy - ixy * ixy)))
    high = np.abs(trace / 2.0 + root)
    low = np.abs(trace / 2.0 - root)
    return np.maximum(0.0, np.maximum(high, low) - np.minimum(high, low))


def brush_ridge(img):
    """
    The score the brush searches with: a fine difference of Gaussians weighted
    by its own line-likeness.

    Taking the Hessian of the DoG, rather than of a separately band-passed
    copy of the image, matters for more than speed. It adds only a 3x3
    neighbourhood, so a padded window still gives bit-identical results to the
    whole image and the preview keeps matching what a save writes.

    The DoG is clipped at zero first, so only features brighter than their
    surroundings count and dark detail is left alone - the same rule the
    signed test used before.
    """
    dog = np.maximum(difference_of_gaussians(img, *BRUSH_SIGMA), 0.0)
    return dog * _line_likeness(dog)


def dust_mask(img, sig_min, sig_max, threshold, spread,
              marks=None, protect=None, want_found=False, auto=True):
    """
    Segment dust particles and return a soft blending weight in [0, 1].

    Dust shows up as small bright specks, so we intersect a difference of
    Gaussians (which responds to features between sig_min and sig_max) with a
    plain brightness threshold and with a test that something is actually
    sitting there (see `_stands_proud`), then expand the result to cover the
    particle's soft edges.

    `threshold` is a percentage of full scale, so one setting means the same
    thing whether the scan is 8- or 16-bit.

    Inside a brushed region the rules change. Painting is the user asserting
    "there is something here", so the brightness test is dropped entirely and
    the search switches to `brush_ridge`, a much finer band weighted by how
    line-like each pixel is. That is what lets a hair far too faint to clear
    the global threshold still be found. With `force` the search is skipped and
    the stroke itself is taken as the answer, for marks no detector will agree
    to.

    `protect` wins over everything, and is applied after expansion so a nearby
    repair cannot bleed into it.
    """
    soft, _particles, idle, _groups = _segment(img, sig_min, sig_max, threshold,
                                               spread, marks, protect, auto)
    if want_found:
        return soft, idle
    return soft


def _stands_proud(img):
    """
    True where a pixel is brighter than its surroundings, not merely bright.

    The brightness test alone cannot tell dust from the sky beside a dark wire,
    because that sky really is bright - brighter, often, than dust elsewhere on
    the frame. What separates them is whether anything is actually sitting
    there. An opening erases bright features smaller than its structuring
    element and leaves everything else alone, so the height above it is the
    speck's own height, and is near zero on an ordinary stretch of sky however
    bright that sky is or whatever it sits next to.
    """
    height = white_tophat(img, footprint=TOPHAT_DISK)
    return height > TOPHAT_LEVEL * full_scale(img.dtype)


def _segment(img, sig_min, sig_max, threshold, spread,
             marks=None, protect=None, auto=True):
    """
    Shared segmentation.

    Returns the soft blending weight, the tight pre-expansion particle mask,
    and the marked-but-empty region. The tight mask matters to the filler: how
    far the median can see is set by the particle's real size, not by how far
    the mask was afterwards expanded.

    `marks` is a list of the user's own marks, each carrying its own settings
    rather than sharing one global pair. That matters because dust does not
    arrive in one flavour: a dense speck on sky wants a different sensitivity
    from a faint hair over foliage, and one slider for all of them means every
    mark is tuned by whichever was adjusted last. Each entry is a dict with

        mask         where the user marked, as a boolean array
        sensitivity  the ridge cut for this mark alone
        force        take the mark as the answer, skipping the search entirely

    A spot is simply a mark with `force` set: you clicked on a speck and said
    repair that, so there is nothing to search for.
    """
    # Every contributor is a part: what it claims, how far that claim is
    # feathered, and how the hole should be filled. The global pass is just the
    # first part rather than a special case, which is what lets a mark carry
    # its own spread and fill without the two paths diverging.
    parts = []
    if auto:
        signed = difference_of_gaussians(img, sig_min, sig_max)
        bright = img > (threshold / 100.0) * full_scale(img.dtype)
        core = np.logical_and(bright, np.abs(signed) > DOG_LEVEL)
        parts.append((np.logical_and(core, _stands_proud(img)), spread, None))

    marked = None
    live = [m for m in (marks or []) if m["mask"].any()]
    if live:
        # The ridge is the expensive part and it does not depend on any mark's
        # settings, so it is computed once however many marks there are and
        # only when at least one is searching.
        ridge = None
        if any(not m.get("force") for m in live):
            ridge = brush_ridge(img)
        marked = np.zeros(img.shape, dtype=bool)
        for mark in live:
            marked |= mark["mask"]
            if mark.get("force"):
                got = mark["mask"]
            else:
                level = brush_level(mark.get("sensitivity",
                                             BRUSH_DEFAULTS["sensitivity"]))
                got = np.logical_and(mark["mask"], ridge > level)
            parts.append((got, mark.get("spread") or spread, mark.get("fill")))

    combined = np.zeros(img.shape, dtype=bool)
    soft = np.zeros(img.shape, dtype=np.float64)
    groups = []
    for core, reach, how in parts:
        if not core.any():
            continue
        combined |= core
        # Feathered per part, then combined by taking whichever claims a pixel
        # most strongly. Expanding the union once instead would give every part
        # the same reach, which is the thing being fixed.
        grown = _expand(core, reach)
        soft = np.maximum(soft, grown)
        # The group is the *expanded* claim, not the tight core. A group says
        # which method fills a pixel, so it has to cover every pixel that will
        # be filled; handing over the tight core instead leaves the feathered
        # ring unfilled, which is a different picture and not the one the
        # window and the save agree on.
        groups.append((grown > 0.5, how))

    if protect is not None and protect.any():
        soft = soft * ~protect
        combined = np.logical_and(combined, ~protect)
        groups = [(np.logical_and(c, ~protect), how) for c, how in groups]

    # Where the user marked but nothing cleared the bar - shown in the mask
    # view so a mark that did nothing is visible rather than silent.
    idle = (np.logical_and(marked, ~combined) if marked is not None else None)
    return soft, combined, idle, groups


def _regional_median(img, radius, mask):
    """
    Median of each pixel's masked neighbourhood.

    16-bit data is binned down to MEDIAN_BITS first, because the rank filter
    cost scales with the number of histogram bins. The result is stretched back
    over the full range, so the repaired pixels land within about 0.02% of full
    scale of the true median.
    """
    footprint = disk(radius)
    if img.dtype != np.uint16:
        return rank_median(img, footprint=footprint, mask=mask)

    shift = 16 - MEDIAN_BITS
    top = (1 << MEDIAN_BITS) - 1
    binned = (img >> shift).astype(np.uint16)
    out = rank_median(binned, footprint=footprint, mask=mask)
    return (out.astype(np.uint32) * 65535 // top).astype(np.uint16)


def _beyond_median_reach(core, soft, intensity):
    """
    The parts of the mask the median cannot repair.

    Two things have to be true at once.

    First, the median has to be losing. `_regional_median` is given `soft > 0`
    as its sample pool - the dust plus the feathered halo around it - and a
    median returns dust exactly when more than half of that pool is dust. No
    tuned cut is involved there; the half is the definition of a median. This
    is also why the median works at all on a small speck: its pool is never
    clean film, it is the speck plus a halo that is mostly clean, and the halo
    outvotes the speck until the speck grows too big.

    Second, the mask has to be genuinely thick. The vote alone assumes that
    masked means dust, and on a real scan much of the mask is not dust at all -
    the difference of Gaussians leaves thin ribbons along edges - so on its own
    it hands about 80% of a frame to the solver, which costs grain everywhere
    and, because each biharmonic solve is global to its region, breaks the
    preview's agreement with the save. Requiring thickness as well drops that
    to about half while measuring better on every speck size tested.

    The decision stays local either way: a box of radius `intensity` for the
    vote, and a thickness test that seeds on depth and grows back by the same
    amount. Handing over whole connected regions would be tidier, since then
    the two fills could never meet inside a single repair, but a region can
    sprawl for thousands of pixels across a scan - much further than a preview
    window can see - and preview and save then disagree about it.
    """
    # A box rather than the filter's disc: separable, so the cost does not grow
    # with the radius, and the vote it takes is within a percent of the disc's.
    size = 2 * intensity + 1
    dust = uniform_filter(core.astype(np.float32), size)
    pool = uniform_filter((soft > 0).astype(np.float32), size)
    outvoted = np.logical_and(core, dust > 0.5 * pool)
    if not outvoted.any():
        return np.zeros_like(core)

    # Thicker than the median's own reach. Expansion has already grown the mask
    # past the particle, so a rim counts only via the dilation back out.
    reach = intensity + 1
    seeds = distance_transform_edt(binary_fill_holes(core)) > reach
    if not seeds.any():
        return np.zeros_like(core)
    thick = distance_transform_edt(~seeds) <= reach
    return np.logical_and(outvoted, thick)


def _grain_energy(values):
    """
    How much fine detail sits at each pixel - grain, mostly.

    A centred difference in each direction, which is the same quantity
    DarkSlide scores its donors on. It is deliberately not a band-pass: what
    the donor search needs is a local measure of how busy the film is, and the
    cheapest honest one is how far neighbouring pixels sit apart.
    """
    gy = np.empty_like(values)
    gx = np.empty_like(values)
    gy[1:-1] = values[2:] - values[:-2]
    gy[0], gy[-1] = gy[1], gy[-2]
    gx[:, 1:-1] = values[:, 2:] - values[:, :-2]
    gx[:, 0], gx[:, -1] = gx[:, 1], gx[:, -2]
    return np.hypot(gx, gy)


_RING_CACHE = {}
# The sixteen probe directions, as unit vectors. Recomputing them per blob cost
# 92,000 trig calls on a 12 megapixel scan for sixteen distinct answers.
_PROBE = tuple((np.sin(2.0 * np.pi * s / 16.0), np.cos(2.0 * np.pi * s / 16.0))
               for s in range(16))


def _ring_mask(radius, inner):
    """
    The disc, or annulus, a patch is measured over.

    Cached on its two radii. The shape depends on nothing else - the clipping
    against the frame edge is a slice of this, not a different mask - and a
    frame's blobs take only a few dozen distinct sizes between them, so the
    same handful of masks were being rebuilt tens of thousands of times.
    """
    got = _RING_CACHE.get((radius, inner))
    if got is None:
        span = np.arange(-radius, radius + 1)
        dist = span[:, None] ** 2 + span[None, :] ** 2
        got = np.logical_and(dist <= radius * radius, dist >= inner * inner)
        _RING_CACHE[(radius, inner)] = got
    return got


def _patch_stats(values, grain, clean, cy, cx, radius, inner=0):
    """Mean tone and mean grain energy over a disc, ignoring masked pixels."""
    h, w = values.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return None
    ring = _ring_mask(radius, inner)[y0 - cy + radius:y1 - cy + radius,
                                     x0 - cx + radius:x1 - cx + radius]
    use = np.logical_and(ring, clean[y0:y1, x0:x1])
    if int(use.sum()) < 8:
        return None                 # too little clean film to say anything
    return (float(values[y0:y1, x0:x1][use].mean()),
            float(grain[y0:y1, x0:x1][use].mean()))


def _transplant_grain(img, out, region, soft):
    """
    Put real grain back into a solved repair.

    The solver leaves a hole smooth. For each blob it fills, this finds a patch
    of clean film nearby whose tone and grain energy match the ring around that
    blob, and adds that patch's *deviation from its own mean* on top of the
    solved values. Subtracting the donor's mean is what lets grain from film of
    one brightness sit correctly on a repair of another - only the texture
    travels, never the tone.

    A donor that is itself masked would carry dust into the repair, so those
    pixels are skipped and left as the solver had them.

    The residual is a *high-pass* of the donor rather than its deviation from
    the patch mean. Subtracting a mean is a box high-pass as wide as the patch,
    which leaves any structure smaller than that in the residual - and a donor
    sitting across a hard edge then carries half the edge contrast into the
    repair. Measured on a hair lying along the horizon of a real scan, that put
    a bright wedge into the sky that no other fill produced. Removing
    everything coarser than the grain first leaves only what this is for.
    """
    values = img.astype(np.float64)
    grain = _grain_energy(values)
    # Grain is a one-pixel feature; anything broader than a few pixels is the
    # picture, and must not travel with it.
    detail = values - gaussian(values, sigma=TEXTURE_GRAIN_SIGMA,
                               preserve_range=True)
    clean = ~(soft > 0)             # anything being repaired is not a donor
    marks, count = label(region)
    if not count:
        return out
    h, w = img.shape

    for index, box in enumerate(find_objects(marks), start=1):
        if box is None:
            continue
        blob = np.logical_and(marks[box] == index, region[box])
        if not blob.any():
            continue
        span = max(box[0].stop - box[0].start, box[1].stop - box[1].start)
        radius = max(2, int(round(span / 2.0)))
        patch = min(radius, TEXTURE_PATCH)
        cy = (box[0].start + box[0].stop) // 2
        cx = (box[1].start + box[1].stop) // 2

        # What the repair should end up looking like: the clean ring around it.
        want = _patch_stats(values, grain, clean, cy, cx, min(radius * 2, 48),
                            inner=radius)
        if want is None:
            continue

        # `away` is clamped at TEXTURE_REACH, so on any blob of radius 15 or
        # more all three multiples give the same sixteen offsets and two thirds
        # of the probes repeat. Measured on a 12 megapixel scan, 10% of 46,176
        # probes were exact duplicates. Dropping them cannot change the answer:
        # the winner is taken on a strict `<`, so a repeat could only ever tie
        # with the probe it repeats and would never displace it.
        seen = set()
        offsets = []
        for multiple in (2.1, 2.9, 3.9):
            away = min(multiple * radius, float(TEXTURE_REACH))
            for sin_a, cos_a in _PROBE:
                step = (int(round(sin_a * away)), int(round(cos_a * away)))
                if step not in seen:
                    seen.add(step)
                    offsets.append(step)

        best = None
        for dy, dx in offsets:
            got = _patch_stats(values, grain, clean, cy + dy, cx + dx, patch)
            if got is None:
                continue
            score = (abs(got[0] - want[0])
                     + TEXTURE_GRAIN_WEIGHT * abs(got[1] - want[1]))
            if best is None or score < best[0]:
                best = (score, dy, dx)
        if best is None:
            continue                # nowhere clean to copy from; stays smooth

        _score, dy, dx = best
        ys, xs = np.where(blob)
        ys = ys + box[0].start
        xs = xs + box[1].start
        sy = np.clip(ys + dy, 0, h - 1)
        sx = np.clip(xs + dx, 0, w - 1)
        usable = clean[sy, sx]
        out[ys[usable], xs[usable]] += detail[sy[usable], sx[usable]]
    return out


def _fill_regions(img, intensity, soft, method, groups=None):
    """
    Replacement values for the masked area, as float.

    The median copies real neighbouring pixels, so it keeps film grain and is
    the better choice on small particles. The biharmonic solver continues the
    surrounding gradients across the hole instead, which is what large
    particles need. `auto` uses the median except where it cannot reach.

    `groups` lets different parts of one mask be filled differently - a hair
    over foliage rebuilt while a speck on sky keeps its grain. Each entry is
    (core, method), and a method of None means fall back to `method`. Later
    groups win where they overlap, so a mark placed on top of a global find
    decides how that pixel is repaired.

    The median is computed once for the whole frame however many groups ask for
    it, because it is the expensive one and its sample pool is the same either
    way: `soft > 0` - every mask's feathered halo, which is the clean film.
    """
    core = soft > 0.5
    if not core.any():
        return img.astype(np.float64)
    if groups is None:
        groups = [(core, method)]

    out = img.astype(np.float64)
    top = full_scale(img.dtype)
    median = None
    solve = np.zeros_like(core)
    textured = np.zeros_like(core)      # solved, then given its grain back

    for g_core, g_method in groups:
        g_core = np.logical_and(g_core, core)
        if not g_core.any():
            continue
        how = method if g_method is None else g_method
        if how in (FILL_BIHARMONIC, FILL_TEXTURE):
            solve |= g_core
            # Texture is Smooth plus a transplant, so both reach the solver and
            # only one of them comes back for grain. A later group overrules an
            # earlier one here as everywhere else.
            if how == FILL_TEXTURE:
                textured |= g_core
            else:
                textured &= ~g_core
            continue
        if median is None:
            # Left in the source dtype. Assigning into a float64 array converts
            # exactly, so promoting the whole frame first only bought a 94 MB
            # copy of which under 1% is ever read.
            median = _regional_median(img, intensity, soft > 0)
        out[g_core] = median[g_core]
        solve &= ~g_core                # a later group overrules an earlier one
        textured &= ~g_core
        if how == FILL_AUTO:
            solve |= _beyond_median_reach(g_core, soft, intensity)

    if solve.any():
        # The solver continues the surrounding gradients across a hole, so it
        # needs surroundings: given a region with no unmasked pixel anywhere to
        # read from, it reduces over an empty set and raises. That cannot
        # happen at an ordinary dust load, but a blown-out frame with global
        # detection on can reach it, and a crash mid-render is a bad way to
        # find out. With nothing to continue from, the honest answer is to
        # leave those pixels as they are.
        if solve.all():
            return out
        try:
            solved = inpaint_biharmonic(img.astype(np.float64) / top, solve)
        except ValueError:
            return out
        # Rescaled after the selection rather than before it. Multiplying is
        # elementwise, so the values are identical either way, but this way the
        # multiply runs over the mask instead of over the frame.
        out[solve] = solved[solve] * top
        if textured.any():
            out = _transplant_grain(img, out, textured, soft)
    return out


def remove_dust(img, intensity, sig_min, sig_max, threshold, spread,
                marks=None, protect=None, want_found=False, fill=DEFAULT_FILL,
                auto=True):
    """Replace segmented dust with a reconstruction of its surroundings."""
    mask, _particles, idle, groups = _segment(img, sig_min, sig_max, threshold,
                                              spread, marks, protect, auto)

    filled = _fill_regions(img, intensity, mask, fill, groups)
    # Blend only where the mask actually claims something. Outside it the sum
    # is `filled * 0 + img * 1`, which is the original pixel exactly, and it
    # cannot leave the dtype's range - so the six full-frame float64 arrays the
    # unrestricted expression allocates are all spent reproducing `img`. On a
    # 12 MP scan the mask is about 1% of the frame.
    cleaned = img.copy()
    touch = mask > 0
    if touch.any():
        weight = mask[touch]
        blended = filled[touch] * weight + img[touch] * (1 - weight)
        cleaned[touch] = np.clip(blended, 0, full_scale(img.dtype)
                                 ).astype(img.dtype)
    if want_found:
        return cleaned, mask, idle
    return cleaned, mask


def algo_args(params):
    """
    The part of a params dict `remove_dust` accepts, as keyword arguments.

    There are two callers - the windowed preview and the full render - and for
    the preview to be honest they must pass the algorithm exactly the same
    settings. Written out twice, they drift: a control added to one and not the
    other is a preview that disagrees with the file on disk, silently.

    Deriving the set from `VIEW_ONLY` rather than listing it also means a new
    slider is handled by adding it to `DEFAULTS`, and a new *view* control by
    adding it to `VIEW_ONLY`, with nothing here to keep in step. `bench.py`
    once filtered on the literal `"window"` instead, and when Division arrived
    it passed `division=` straight into `remove_dust` and raised.
    """
    args = {k: v for k, v in params.items() if k not in VIEW_ONLY}
    args.setdefault("fill", DEFAULT_FILL)
    args.setdefault("auto", True)
    return args


def widest_spread(spread, marks=None):
    """The largest feather any part of the mask will use."""
    return max([spread] + [m.get("spread") or spread for m in (marks or [])])


def uses_texture(fill, marks=None):
    """Whether anything in this frame will ask for a grain transplant."""
    return (fill == FILL_TEXTURE
            or any(m.get("fill") == FILL_TEXTURE for m in (marks or [])))


def _margin_for(sig_max, spread, intensity, texture=False):
    """
    How much context a window needs so its result matches the full image.

    Gaussians are truncated at 4 sigma and the median reaches `intensity`
    pixels, so a window padded by this much is unaffected by the crop edge.
    The brush's own band is included in case the global sigmas are set below
    it; the Hessian on top of it reaches only one further pixel.

    The top-hat is an erosion followed by a dilation, so a pixel's height above
    the opening depends on the image two structuring elements away.
    """
    widest = max(sig_max, BRUSH_SIGMA[1])
    margin = int(4 * widest + 4 * spread + intensity + 2 * TOPHAT_RADIUS + 5)
    # The grain transplant reads a donor patch up to TEXTURE_REACH away, so a
    # window has to hold that too or a repair would copy from film the preview
    # cannot see. Added only when something actually asks for it - it is 44 px
    # on every side, and paying that on every preview would be a real cost.
    if texture:
        margin += TEXTURE_REACH + TEXTURE_PATCH
    return margin


def slice_marks(marks, y0, x0, y1, x1):
    """
    The user's marks restricted to one region, still carrying their settings.

    Marks are stored as a small array plus an origin rather than a full-frame
    layer: a 12 megapixel boolean is 13 MB, and a working session can leave
    dozens of them, so keeping each at its own size is the difference between
    a few kilobytes and a gigabyte.
    """
    out = []
    for mark in marks or []:
        my, mx = mark["y"], mark["x"]
        mh, mw = mark["mask"].shape
        ty0, tx0 = max(y0, my), max(x0, mx)
        ty1, tx1 = min(y1, my + mh), min(x1, mx + mw)
        if ty0 >= ty1 or tx0 >= tx1:
            continue                    # this mark is not in this window
        piece = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        piece[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] =             mark["mask"][ty0 - my:ty1 - my, tx0 - mx:tx1 - mx]
        if not piece.any():
            continue
        # Every setting travels with the mark. Dropping any of them here does
        # not fail loudly - the window quietly falls back to the global value
        # while a save uses the mark's own, so the preview stops matching the
        # file for exactly the marks you tuned by hand.
        out.append({"mask": piece, "sensitivity": mark.get("sensitivity"),
                    "force": mark.get("force", False),
                    "spread": mark.get("spread"), "fill": mark.get("fill")})
    return out


def process_window(img, y, x, height, width, params, marks=None, protect=None):
    """
    Run dust removal on one region of the image.

    The region is processed with surrounding context and then trimmed, so what
    the preview shows is what a full-image save produces there. The brush
    layers are sliced with the same padding, so that holds for painted areas
    too.

    The mask matches exactly - segmentation is entirely local, and over a sweep
    of 648 windows the soft weight came back bit-identical every time. So every
    control that decides *what* is repaired previews truthfully.

    The replacement values do not, under Smooth or Texture. `inpaint_biharmonic`
    solves a sparse linear system rather than reading a bounded neighbourhood,
    and the values it returns for one masked region depend on what else is
    masked anywhere in the frame.

    This was recorded for a long time as "a mark running off the padded edge is
    solved against different surroundings", which is true and is not the whole
    story. Measured on a real scan at 3.8% mask load: three regions lying
    entirely inside the padded crop - the furthest 44 px clear of it - differed
    by up to 4225/65535. Masking any one of them alone and inpainting arrays of
    117, 160, 200 and 320 px gives bit-identical pixels, so each region is
    size-independent by itself; adding the frame's other regions back is what
    moves it, with `split_into_regions` either way. So the perturbation travels
    between regions nowhere near each other, and no predicate on the geometry
    predicts it - a guard built on one was written, measured and abandoned.

    What does hold, and what `test_parity.py` asserts: the mask is exact,
    nothing outside the mask moves, and Median is exact. Saving is unaffected -
    it processes the whole frame, so the file on disk is always the correct
    result.

    Left as is deliberately: see the README for the fixes weighed against it.
    A warning was tried and dropped for firing on a quarter of all windows.

    Returns (original, cleaned, mask, idle, protect) crops, each height x width.
    """
    margin = _margin_for(params["sig_max"],
                         widest_spread(params["spread"], marks),
                         params["intensity"],
                         texture=uses_texture(params.get("fill", DEFAULT_FILL),
                                              marks))
    y0, x0 = max(0, y - margin), max(0, x - margin)
    y1 = min(img.shape[0], y + height + margin)
    x1 = min(img.shape[1], x + width + margin)
    pad = (slice(y0, y1), slice(x0, x1))

    sub = img[pad]
    sub_marks = slice_marks(marks, y0, x0, y1, x1)
    sub_protect = protect[pad] if protect is not None else None

    cleaned, mask, idle = remove_dust(
        sub,
        marks=sub_marks,
        protect=sub_protect,
        want_found=True,
        **algo_args(params),
    )

    oy, ox = y - y0, x - x0
    sl = (slice(oy, oy + height), slice(ox, ox + width))
    return (sub[sl], cleaned[sl], mask[sl],
            None if idle is None else idle[sl],
            None if sub_protect is None else sub_protect[sl])


def load_gray(path):
    """
    Load an image as a grayscale array on its own tonal scale.

    The source bit depth is preserved: 8-bit files stay uint8, while anything
    deeper (16-bit TIFF, float TIFF) is carried through as uint16 so archival
    scans keep their tonal range.

    Note this deliberately departs from an earlier approach, which divided by
    the image's own maximum and so stretched every scan to hit pure white. On a
    calibrated scan that is a tonal edit, not a load, and it shows up as a
    brightness shift when the result goes back into an editor.
    """
    img = skio.imread(path)

    # Decide the depth from the file itself, before rgb2gray turns it to float.
    dtype = np.uint16 if img.dtype.itemsize > 1 else np.uint8
    top = full_scale(dtype)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = skimage.color.rgba2rgb(img)
        values = skimage.color.rgb2gray(img) * top      # rgb2gray returns 0..1
    elif img.ndim == 2:
        if img.dtype.kind == "f":
            values = img.astype(np.float64) * top       # float images are 0..1
        else:
            source_top = float(np.iinfo(img.dtype).max)
            values = img.astype(np.float64) * (top / source_top)
    else:
        raise ValueError("Unsupported image with %d dimensions" % img.ndim)

    return np.clip(np.rint(values), 0, top).astype(dtype)


def read_source_meta(path):
    """
    Note how the source file is laid out, so the export can match it.

    Editors are picky in ways the TIFF spec does not require. Capture One in
    particular will not import a single channel greyscale TIFF, so a scan that
    arrived as RGB has to leave as RGB even though the pixels are grey.

    The ICC profile matters most of all. A scan tagged Adobe RGB that is
    written back untagged gets read as sRGB, and the tone curve difference
    between the two shows up as a plain brightness shift.
    """
    meta = {"channels": 1, "resolution": None, "unit": 2, "rows_per_strip": None,
            "icc": None, "orientation": None, "make": None, "model": None,
            "xmp": None}
    lower = path.lower()

    if lower.endswith((".tif", ".tiff")):
        try:
            with tifffile.TiffFile(path) as handle:
                page = handle.pages[0]
                tags = page.tags
                if "SamplesPerPixel" in tags:
                    meta["channels"] = int(tags["SamplesPerPixel"].value)
                elif int(page.photometric) == 2:
                    meta["channels"] = 3
                if "XResolution" in tags and "YResolution" in tags:
                    xr, yr = tags["XResolution"].value, tags["YResolution"].value
                    meta["resolution"] = (xr[0] / xr[1], yr[0] / yr[1])
                if "ResolutionUnit" in tags:
                    meta["unit"] = int(tags["ResolutionUnit"].value)
                if "RowsPerStrip" in tags:
                    meta["rows_per_strip"] = int(tags["RowsPerStrip"].value)
                # Named InterColorProfile in TIFF, tag 34675.
                if "InterColorProfile" in tags:
                    meta["icc"] = bytes(tags["InterColorProfile"].value)
                if "Orientation" in tags:
                    meta["orientation"] = int(tags["Orientation"].value)
                for key in ("Make", "Model"):
                    if key in tags:
                        meta[key.lower()] = str(tags[key].value)
                # Ratings, keywords and labels live here.
                if "XMP" in tags:
                    meta["xmp"] = bytes(tags["XMP"].value)
        except Exception:
            pass                        # metadata is a nicety, never fatal
    else:
        try:
            with Image.open(path) as im:
                dpi = im.info.get("dpi")
                if dpi:
                    meta["resolution"] = (float(dpi[0]), float(dpi[1]))
                meta["icc"] = im.info.get("icc_profile")
        except Exception:
            pass
    return meta


def save_array(arr, path, meta=None):
    """
    Write the result, keeping 16-bit depth wherever the format allows.

    JPEG is 8-bit only, so 16-bit data is stepped down for that format alone;
    TIFF and PNG are written at full depth. TIFF output mirrors the source's
    channel count and resolution so it drops back into the same workflow.
    """
    meta = meta or {}
    lower = path.lower()
    resolution = meta.get("resolution")

    icc = meta.get("icc")

    if lower.endswith((".jpg", ".jpeg")):
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        extra = {"dpi": resolution} if resolution else {}
        if icc:
            extra["icc_profile"] = icc
        Image.fromarray(arr, mode="L").save(path, quality=95, subsampling=0,
                                            **extra)
        return

    if lower.endswith((".tif", ".tiff")):
        data = arr
        if meta.get("channels", 1) >= 3:
            # Grey pixels carried in three channels: same picture, but a
            # layout every editor accepts.
            data = np.repeat(arr[:, :, None], 3, axis=2)

        options = {
            "photometric": "rgb" if data.ndim == 3 else "minisblack",
            "software": "Dust Removal",
            "metadata": None,       # suppress tifffile's own ImageDescription
        }
        if resolution:
            options["resolution"] = resolution
            options["resolutionunit"] = meta.get("unit", 2)
        if meta.get("rows_per_strip"):
            options["rowsperstrip"] = meta["rows_per_strip"]
        if icc:
            options["iccprofile"] = icc

        # 2 = ASCII, 3 = SHORT. Carried so the file still says where it came
        # from and which way up it is.
        extra = []
        if meta.get("orientation"):
            extra.append((274, 3, 1, int(meta["orientation"]), True))
        for code, key in ((271, "make"), (272, "model")):
            if meta.get(key):
                text = meta[key]
                extra.append((code, 2, len(text) + 1, text.encode() + b"\x00", True))
        if meta.get("xmp"):
            extra.append((700, 1, len(meta["xmp"]), meta["xmp"], True))
        if extra:
            options["extratags"] = extra

        tifffile.imwrite(path, data, **options)
        return

    # PNG and anything else. No mode= here: passing one is deprecated, and
    # letting Pillow infer it is what yields a true 16-bit I;16 image.
    image = Image.fromarray(arr)
    extra = {}
    if resolution:
        extra["dpi"] = resolution
    if icc:
        extra["icc_profile"] = icc
    image.save(path, **extra)


def window_origin(shape, h, w, size, whole=False):
    """
    The region to preview, as (y, x, height, width).

    `whole` ignores the position and size entirely and returns the frame, which
    is how Wide View asks for everything at once.
    """
    ih, iw = shape
    if whole:
        return 0, 0, ih, iw
    size = max(1, min(size, ih, iw))
    y = int(round(h * max(0, ih - size)))
    x = int(round(w * max(0, iw - size)))
    return y, x, size, size


# --------------------------------------------------------------------------
# Survey mode - stepping over the frame a section at a time
# --------------------------------------------------------------------------

# How many sections a pass over one frame should come to, before the size
# limits below bite. Panning by hand has no end condition: you stop when you
# get bored, not when the frame is covered, and the corners are what gets
# missed. A numbered list of sections turns that into a finite job.
#
# Twenty-four is picked against how long a section takes to look at, not how
# long it takes to compute - on a 4388x2925 scan it puts a section at 795 px,
# which measures 0.30s with the global pass on and 0.01s with it off, so the
# machine is never what you are waiting for.
SURVEY_SECTIONS = 24
# How far neighbouring sections reach into each other. Without this a speck on
# a seam is half in one section and half in the next, and neither view is the
# one you can judge it in. The detector will not accept anything wider than
# 2 * TOPHAT_RADIUS and a repair spreads a few pixels past that, so 64 px is
# enough that every speck is whole in at least one section.
SURVEY_OVERLAP = 64
# Below this a section stops being a working view and becomes a thumbnail, so
# small images get fewer sections rather than smaller ones.
SURVEY_MIN = 128
# The Loupe display: the same job the Cells thumbnail does for the frame, done
# for the cell you are in. Square, because cells are. The mode row it shares is
# as wide as the thumbnail above it, and the buttons, stepper and slider want
# 141 px of that between them.
SECTION_PANE = 112
# How tall the row above the previews is held, always. It is the taller of the
# two things that can sit in it - the pane titles, and the survey step buttons -
# so that one appearing does not move the other or resize the previews.
PANE_HEADER = 34


def survey_size(shape, sections=SURVEY_SECTIONS):
    """
    How big one section should be, for an image of this shape.

    Sections are square because the preview window is; the frame's aspect ratio
    is carried by the *grid* instead, which is chosen to be as close to that
    ratio as whole numbers allow. A 3:2 frame comes out 6 sections by 4 rather
    than 24 by 1, so a section is a piece of the picture rather than a strip.

    The result is clamped at both ends. WINDOW_MAX is where processing a window
    stops being interactive; SURVEY_MIN is where a section stops being big
    enough to work in. A 96 MP scan therefore gets more than `sections`, and a
    small one fewer - the count follows from the size, never the other way
    round, or a thumbnail would be surveyed in 24 pieces.
    """
    ih, iw = shape
    if ih <= 0 or iw <= 0:
        return SURVEY_MIN
    cols = max(1, int(round(math.sqrt(sections * iw / float(ih)))))
    rows = max(1, int(round(sections / float(cols))))
    side = math.ceil(max(iw / float(cols), ih / float(rows)))
    # Plus the overlap: without it the sections computed here would abut, and
    # the grid below would have to add a row and a column to pull them apart.
    return int(min(WINDOW_MAX, max(SURVEY_MIN, side + SURVEY_OVERLAP)))


def _survey_count(length, size, overlap):
    """
    How many sections of `size` it takes to cover `length` overlapping by at
    least `overlap`.

    n sections spread evenly leave a stride of (length - size) / (n - 1), so
    they overlap by at least `overlap` once n >= (length - overlap) / (size -
    overlap). Solving it this way rather than stepping and checking means the
    sections are evenly spaced - a trailing section flush against the edge
    would overlap its neighbour by whatever was left over, sometimes almost
    entirely.
    """
    if size >= length:
        return 1
    stride = size - overlap
    if stride < 1:                  # only reachable if size <= overlap
        stride = max(1, size // 2)
    return max(1, int(math.ceil((length - overlap) / float(stride))))


def survey_grid(shape, size, overlap=SURVEY_OVERLAP):
    """
    Where every section starts, in reading order, as (origins, rows, cols).
    """
    ih, iw = shape
    size = max(1, min(int(size), ih, iw))
    overlap = max(0, min(int(overlap), size - 1))
    rows = _survey_count(ih, size, overlap)
    cols = _survey_count(iw, size, overlap)
    ys = [0 if rows == 1 else int(round(r * (ih - size) / float(rows - 1)))
          for r in range(rows)]
    xs = [0 if cols == 1 else int(round(c * (iw - size) / float(cols - 1)))
          for c in range(cols)]
    return [(y, x) for y in ys for x in xs], rows, cols


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

# Mask view colours: what gets repaired, what was painted to no effect, and
# what is being protected.
LAYER_COLOURS = (
    ("mask", (255, 60, 60)),
    ("idle", (80, 220, 90)),
    ("protect", (70, 150, 255)),
)


def to_pil(arr, layers=None):
    """
    Grayscale array to a PIL image.

    `layers` is a dict of boolean overlays keyed by the names in
    LAYER_COLOURS, painted on in that order.
    """
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)     # screens are 8-bit anyway
    if not layers:
        return Image.fromarray(arr, mode="L")

    rgb = np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    for name, colour in LAYER_COLOURS:
        hit = layers.get(name)
        if hit is None or not hit.any():
            continue
        for channel in range(3):
            # Tint rather than replace, so texture stays readable underneath.
            blended = 0.35 * rgb[hit, channel] + 0.65 * colour[channel]
            rgb[hit, channel] = blended.astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def stamp(mask, cy, cx, radius, value=True):
    """Paint a filled disc into a boolean mask, clipped at the edges."""
    h, w = mask.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy = np.arange(y0, y1)[:, None] - cy
    xx = np.arange(x0, x1)[None, :] - cx
    disc = (yy * yy + xx * xx) <= radius * radius
    if value:
        mask[y0:y1, x0:x1] |= disc
    else:
        mask[y0:y1, x0:x1] &= ~disc


def stamp_line(mask, y0, x0, y1, x1, radius, value=True):
    """Paint along a segment, so a fast drag leaves no gaps."""
    dy, dx = y1 - y0, x1 - x0
    steps = int(max(abs(dy), abs(dx)) / max(1, radius * 0.4)) + 1
    for i in range(steps + 1):
        t = i / steps
        stamp(mask, int(round(y0 + dy * t)), int(round(x0 + dx * t)), radius, value)


def fit(im, box, shrink=Image.LANCZOS):
    """
    Scale a PIL image to fit a box, using nearest neighbour when zooming in.

    `shrink` is the filter used when the image has to come down. Lanczos suits
    the previews, which shrink by very little. The navigator does not: it takes
    a 12MP scan to about 285 px, and at that reduction only a full area average
    (Image.BOX) removes everything above the thumbnail's own Nyquist limit.
    """
    bw, bh = box
    if bw < 1 or bh < 1:
        return im
    ratio = min(bw / im.width, bh / im.height)
    size = (max(1, int(im.width * ratio)), max(1, int(im.height * ratio)))
    resample = Image.NEAREST if ratio >= 1 else shrink
    return im.resize(size, resample)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

class DustRemovalApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # The restore size, for when the window is un-maximised.
        self.geometry(self._start_geometry())
        self.minsize(900, 620)
        try:
            self.state("zoomed")        # open filling the screen
        except tk.TclError:
            pass                        # not every window manager has it

        self.image = None           # full grayscale uint8/uint16 image
        self.path = None
        self.queue = []             # scans to work through, from Open folder
        self.queue_pos = -1
        # What the last save wrote, so Next can tell finished from unfinished.
        self._saved_seq = None
        self._saved_sig = None
        self.source_meta = {}       # layout of the file we loaded
        self.result = None          # (original, cleaned, mask, idle, protect)
        self.result_origin = None   # (y, x, h, w) the result was taken from
        self.overview = None        # PIL thumbnail of the full image
        self.overview_scale = 1.0

        # Manual retouching. `protect_mask` is one full-image boolean layer,
        # but `marks` are not: each is kept at its own size with its own
        # origin, for the reason `slice_marks` gives. There is no undo history
        # - the Eraser and Delete remove whole marks instead.
        self.marks = []             # every spot and sweep, each with settings
        self._next_mark = 1
        self._binding = False       # loading a mark into the controls
        self.protect_mask = None
        self._stroke = None         # stroke in progress
        self._last_point = None
        self._spot = None           # a Spot being sized by the drag
        self._pending_fill = DEFAULT_FILL    # what the next mask is born with

        self._photos = {}           # keep PhotoImage references alive
        self._view = None           # how the crop maps onto the preview canvas
        self._scrollers = []        # (canvas, inner) for each scrolling panel
        self.slider_parts = {}      # key -> widgets, so a slider can be greyed
        # The Window size rows in the two tabs, with the widget each one
        # follows, so survey mode can take them off screen and put them back
        # in place. Navigator replaces them there, on the same variable.
        self.window_rows = []
        self._backdrops = {}        # index -> tinted art, or None if absent
        self._tiles = {}            # the last pane-sized backdrop, cached

        # The final render: one whole-frame result, inspected and then saved.
        self.render = None
        self.render_mask = None
        self.render_idle = None
        self.render_sig = None
        self._rendering = False
        self._edit_seq = 0          # bumped by anything touching the brush layers
        self._preview_job = None
        self._drain_job = None
        self._overview_job = None
        self._seq = 0
        self._closing = False
        self._saving = False
        self._close_when_done = False   # asked to quit while a save was running
        self._label_snapshot = None     # settings a save was started with
        self._keys_window = None
        self.keys = load_keys()
        self._section_view = None   # where the section pane drew its crop
        self._surveyed = False      # has this frame been surveyed yet
        self._settling = False      # true while a mode change relays out
        self.rotated = False        # portrait scan turned on its side
        self.source_shape = None    # its shape before that
        self._division_shown = None # the division Navigator was last full at

        # Single background worker; only the newest request matters. Tk is not
        # thread safe, so the worker hands results back through a queue that
        # the main thread drains, rather than touching widgets itself.
        self._req = None
        self._req_lock = threading.Lock()
        self._req_ready = threading.Event()
        self._results = queue.Queue()
        threading.Thread(target=self._worker_loop, daemon=True).start()

        self._apply_theme()
        try:
            # `default` so message boxes and dialogs inherit it too. Cosmetic,
            # so a missing or unreadable icon must not stop the app opening.
            self.iconbitmap(default=resource_path("assets", "app.ico"))
        except tk.TclError:
            pass
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._apply_keys()
        self._set_controls_state("disabled")
        self._drain_results()

    # -- look ---------------------------------------------------------------

    def _start_geometry(self):
        """
        Open big enough that the Detect panel does not start out scrolled.

        Six sliders with an explanation apiece want about 560 px of sidebar.
        The old fixed 1180x780 gave them 346, so the panel opened with a third
        of itself already out of view. Measured against the screen rather than
        hard coded, and clamped at both ends so it stays a window on a large
        monitor and still fits a laptop.
        """
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = max(900, min(1280, sw - 120))
        h = max(620, min(1080, sh - 140))
        return "%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2),
                                max(0, (sh - h) // 2 - 20))

    def _apply_theme(self):
        """
        Pastel styling drawn from the artwork in `assets`.

        `clam` is the one bundled ttk theme that actually honours colour on
        Windows; the native themes draw most widgets from OS bitmaps and quietly
        ignore what they are told.
        """
        t = THEME
        self.configure(bg=t["bg"])
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=t["bg"], foreground=t["ink"],
                        fieldbackground=t["raised"], font=FONT,
                        borderwidth=0, focuscolor=t["accent"])
        style.configure("TFrame", background=t["bg"])
        style.configure("TLabel", background=t["bg"], foreground=t["ink"])
        style.configure("Hint.TLabel", foreground=t["faint"], font=FONT_SMALL)
        style.configure("Heading.TLabel", font=FONT_BOLD)
        style.configure("Status.TLabel", background=t["raised"],
                        foreground=t["faint"], font=FONT_SMALL,
                        relief="flat", padding=(10, 5))
        # The strip the two status labels sit on, so the gap between them is
        # the same colour as they are rather than the window showing through.
        style.configure("Status.TFrame", background=t["raised"])

        # White on the plum, which measures 5.6:1. The accent was darkened a
        # little from the original for exactly this - at its first value white
        # sat at 4.29:1, just under what 9pt bold needs.
        style.configure("TButton", background=t["accent"], foreground="#ffffff",
                        font=FONT_BOLD, relief="flat", padding=(10, 5))
        # A disabled button used to be drawn in a cold blue-grey left over from
        # another palette, which read as a different application's widget.
        style.map("TButton",
                  background=[("pressed", t["accent_dark"]),
                              ("active", t["accent_dark"]),
                              ("disabled", "#e6dee6")],
                  foreground=[("disabled", "#a794a7")])

        for kind in ("TCheckbutton", "TRadiobutton"):
            style.configure(kind, background=t["bg"], foreground=t["ink"],
                            indicatorcolor=t["raised"], focusthickness=0)
            style.map(kind,
                      background=[("active", t["bg"])],
                      indicatorcolor=[("selected", t["accent_dark"]),
                                      ("pressed", t["accent"])],
                      foreground=[("disabled", t["faint"])])

        # Survey mode is a mode, not another view option, so its control is
        # drawn at the weight of a toolbar button with the tick still in it -
        # the same padding and bold face as "Open Image...", and the accent
        # fill when it is on, so the state reads from across the room. White on
        # the accent is the 5.59:1 pairing already measured for buttons; the
        # unselected fill is the tab colour, which carries ink at 11.3:1.
        style.configure("Big.TCheckbutton", background=t["accent_soft"],
                        foreground=t["ink"], font=FONT_BOLD, padding=(8, 6),
                        indicatorcolor=t["raised"], focusthickness=0,
                        bordercolor=t["edge"], borderwidth=1, relief="flat")
        style.map("Big.TCheckbutton",
                  background=[("selected", "active", t["accent_dark"]),
                              ("selected", t["accent"]),
                              ("active", t["raised"])],
                  foreground=[("selected", "#ffffff"),
                              ("disabled", t["faint"])],
                  indicatorcolor=[("selected", t["raised"]),
                                  ("pressed", t["accent_soft"])])

        style.configure("Horizontal.TScale", background=t["bg"],
                        troughcolor=t["accent_soft"], borderwidth=0,
                        lightcolor=t["accent"], darkcolor=t["accent"])
        style.map("Horizontal.TScale",
                  background=[("active", t["bg"])])

        style.configure("Vertical.TScrollbar", background=t["accent_soft"],
                        troughcolor=t["bg"], bordercolor=t["bg"],
                        arrowcolor=t["accent_dark"], borderwidth=0,
                        arrowsize=11, width=11)
        style.map("Vertical.TScrollbar",
                  background=[("active", t["accent"]),
                              ("pressed", t["accent_dark"])])

        # A readonly combobox draws its value in the entry's *selection*
        # colours whenever it holds focus, so the value sat in a block of
        # system blue on a control nobody was editing - and the blue came from
        # the OS, not from this palette. Mapping the selection colours onto the
        # field's own makes the value read as plain text.
        style.configure("TCombobox", fieldbackground=t["raised"],
                        background=t["raised"], foreground=t["ink"],
                        selectbackground=t["raised"], selectforeground=t["ink"],
                        arrowcolor=t["accent_dark"], borderwidth=1,
                        lightcolor=t["edge"], darkcolor=t["edge"],
                        bordercolor=t["edge"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", t["raised"]),
                                   ("disabled", t["bg"])],
                  selectbackground=[("readonly", t["raised"])],
                  selectforeground=[("readonly", t["ink"])],
                  foreground=[("disabled", t["dim"])],
                  arrowcolor=[("disabled", t["dim"])])
        # The drop-down list is a plain Tk listbox rather than a themed widget,
        # so it is reached through the option database or not at all.
        self.option_add("*TCombobox*Listbox.background", t["raised"])
        self.option_add("*TCombobox*Listbox.foreground", t["ink"])
        self.option_add("*TCombobox*Listbox.selectBackground", t["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.option_add("*TCombobox*Listbox.font", FONT)

        style.configure("TNotebook", background=t["bg"], borderwidth=0,
                        tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", background=t["accent_soft"],
                        foreground=t["faint"], font=FONT_BOLD,
                        padding=(14, 6), borderwidth=0)
        # The selected tab is drawn larger as well as darker. Colour alone puts
        # the whole distinction on one channel; size is the redundant cue.
        style.map("TNotebook.Tab",
                  background=[("selected", t["accent"])],
                  foreground=[("selected", "#ffffff")],
                  padding=[("selected", (18, 9))],
                  font=[("selected", FONT_TAB)])

    def _pane_art(self, index):
        """
        One pane's artwork, tinted down far enough to sit under the interface.

        Composited onto the panel colour rather than kept transparent. The
        picture fills its pane exactly, so none of this shows at the edges, but
        any transparency in the source still has to land on something.

        Both ends of the ramp are pulled towards that colour, which keeps the
        darkest ink lighter than the lightest text over it - the interface has
        to read as the layer on top.
        """
        if index in self._backdrops:
            return self._backdrops[index]
        self._backdrops[index] = None
        try:
            art = Image.open(resource_path("assets", BACKDROP_FILES[index]))
        except Exception:
            return None                 # decoration; never worth failing over

        v = np.asarray(art.convert("RGBA")).astype(np.float64)
        lum = v[..., :3].mean(2)[..., None] / 255.0     # 0 ink, 1 background
        cover = v[..., 3:4] / 255.0
        base = np.array(_rgb(THEME["canvas"]), dtype=np.float64)
        dark = base + (np.array(_rgb(THEME["ink"]), dtype=np.float64) - base) \
            * BACKDROP_STRENGTH
        light = base + (255.0 - base) * BACKDROP_STRENGTH * 0.35
        tinted = dark + (light - dark) * lum
        out = base * (1.0 - cover) + tinted * cover
        self._backdrops[index] = Image.fromarray(out.round().astype(np.uint8),
                                                 "RGB")
        return self._backdrops[index]

    def _pane_backdrop(self, index, width, height):
        """
        The artwork filling one pane exactly, edge to edge.

        Scaled to *cover* and centre-cropped rather than fitted inside. Fitting
        leaves a margin of panel colour, and fading the picture out into that
        margin is worse still - the eye reads the gradient as the picture being
        unfinished. A pane is a rectangle, so the picture is a rectangle, and
        the only edge is the pane's own border.
        """
        art = self._pane_art(index)
        if art is None or width < 5 or height < 5:
            return None
        key = (index, width, height)
        if key in self._tiles:
            return self._tiles[key]

        ratio = max(width / art.width, height / art.height)
        wide = max(width, int(art.width * ratio + 0.5))
        tall = max(height, int(art.height * ratio + 0.5))
        shrink = Image.BOX if tall < art.height else Image.LANCZOS
        scaled = art.resize((wide, tall), shrink)
        left = (wide - width) // 2
        top = (tall - height) // 2
        self._tiles = {key: scaled.crop((left, top, left + width, top + height))}
        return self._tiles[key]

    def _show_welcome(self):
        """
        The empty state: one picture filling each pane, and one caption.

        Neither pane carries a label inside it - with nothing loaded there is
        no original to label, and a word floating over a picture only asked to
        be read.
        """
        # Which panes are in the layout is a thing this class decides, so ask
        # `single_var` rather than Tk. <Configure> arrives while geometry is
        # still propagating, and a canvas at its final size can still report
        # winfo_ismapped() false - after which no further event comes to
        # correct it, and half the backdrop never gets drawn.
        panes = [self.before_canvas]
        if not self.single_var.get():
            panes.append(self.after_canvas)
        panes = [c for c in panes
                 if c.winfo_width() > 5 and c.winfo_height() > 5]
        for canvas in panes:
            canvas.delete("all")
        if not panes:
            return

        for i, canvas in enumerate(panes):
            tile = self._pane_backdrop(i, canvas.winfo_width(),
                                       canvas.winfo_height())
            if tile is None:
                continue
            key = "backdrop-%d" % i
            self._photos[key] = ImageTk.PhotoImage(tile)
            canvas.create_image(0, 0, anchor="nw", image=self._photos[key])

        last = panes[-1]
        last.create_text(last.winfo_width() // 2, last.winfo_height() - 34,
                         text="Open a scan to begin  -  Ctrl+O",
                         fill=THEME["ink"], font=FONT_BOLD)

    # -- layout ------------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        # Row 1 is the survey strip, which is empty until survey mode is on;
        # the previews take everything else.
        self.rowconfigure(2, weight=1)

        # More room above than below, so the row of buttons is not read as part
        # of the title bar it sits directly under.
        bar = ttk.Frame(self, padding=(10, 14, 10, 8))
        bar.grid(row=0, column=0, sticky="ew")
        self.open_btn = ttk.Button(bar, text="Open image...", command=self.open_image)
        self.open_btn.pack(side="left")
        self.folder_btn = ttk.Button(bar, text="Open folder...",
                                     command=self.open_folder)
        self.folder_btn.pack(side="left", padx=(6, 0))
        self.render_btn = ttk.Button(bar, text="Final render",
                                     command=self.render_full)
        self.render_btn.pack(side="left", padx=(6, 0))
        self.save_btn = ttk.Button(bar, text="Save result...", command=self.save_image)
        self.save_btn.pack(side="left", padx=(6, 0))

        # The queue's own strip, shown only while a folder is open. Next stays
        # greyed while the current scan has work that was never saved; Skip is
        # the deliberate way past it, so nothing is discarded by reflex.
        self.queue_bar = ttk.Frame(bar)
        self.queue_label = ttk.Label(self.queue_bar, text="", font=FONT_BOLD)
        self.queue_label.pack(side="left", padx=(0, 8))
        self.queue_prev = ttk.Button(self.queue_bar, text="Previous",
                                     command=lambda: self.queue_step(-1))
        self.queue_prev.pack(side="left")
        self.queue_skip = ttk.Button(self.queue_bar, text="Skip",
                                     command=lambda: self.queue_step(1, skip=True))
        self.queue_skip.pack(side="left", padx=(6, 0))
        self.queue_next = ttk.Button(self.queue_bar, text="Next",
                                     command=lambda: self.queue_step(1))
        self.queue_next.pack(side="left", padx=(6, 0))

        # The margin left of the previews is the same as the one between them
        # and the sidebar, which is the sidebar's own left padding.
        body = ttk.Frame(self, padding=(SIDE_GAP, 0, SIDE_GAP, 6))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        panes = ttk.Frame(body)
        panes.grid(row=0, column=0, sticky="nsew")
        panes.rowconfigure(1, weight=1)
        # `uniform` as well as `weight`: without it the columns are only
        # equal in the space they *share*, and their minimums come from the
        # titles - "Dust removed" is wider than "Original", which made the
        # right pane 34 px wider than the left and the margins beside them
        # unequal.
        panes.columnconfigure(0, weight=1, uniform="pane")
        panes.columnconfigure(1, weight=1, uniform="pane")

        # The header row is held at the height of the step buttons whether they
        # are shown or not. It costs about ten pixels of preview and it is what
        # makes entering and leaving survey mode change nothing at all in the
        # part of the window being looked at: with the row free to resize, the
        # panes jumped 34 px and the whole image rescaled under the pointer.
        panes.rowconfigure(0, minsize=PANE_HEADER)

        # Filling the row and centring the text inside it, rather than being
        # centred by the grid with a bottom pad: the pad put the titles two
        # pixels above the step buttons placed at the row's middle, which is
        # exactly the sort of near-alignment that reads as a mistake.
        self.before_title = ttk.Label(panes, text="Original", anchor="center",
                                      style="Heading.TLabel")
        self.before_title.grid(row=0, column=0, sticky="nsew")
        self.result_title = ttk.Label(panes, text="Dust removed",
                                      anchor="center", style="Heading.TLabel")
        self.result_title.grid(row=0, column=1, sticky="nsew")

        # Stepping is the one control used over and over while looking at the
        # picture, so it sits with the picture. Placed at the middle of the
        # panes rather than of the window: the sidebar is 300 px of the width,
        # so window-centred put it visibly left of the seam it points at. The
        # two titles are centred in their own columns, which leaves the middle
        # of the header empty for it.
        self.survey_bar = ttk.Frame(panes)
        self.survey_prev = ttk.Button(self.survey_bar, text="◀", width=3,
                                      command=lambda: self._step_survey(-1))
        self.survey_prev.pack(side="left")
        self.survey_label = ttk.Label(self.survey_bar, text="", width=16,
                                      anchor="center", font=FONT_BOLD)
        self.survey_label.pack(side="left", padx=(6, 6))
        self.survey_next = ttk.Button(self.survey_bar, text="▶", width=3,
                                      command=lambda: self._step_survey(1))
        self.survey_next.pack(side="left")

        self.before_canvas = self._make_canvas(panes, 0)
        self.after_canvas = self._make_canvas(panes, 1)

        self._build_sidebar(body)

        # Which file, and what the window is doing with it, on one line at the
        # bottom. The filename used to sit top right, a whole window away from
        # the only other text of its kind - two places to look for one kind of
        # information.
        footer = ttk.Frame(self, style="Status.TFrame")
        footer.grid(row=3, column=0, sticky="ew")
        self.status = ttk.Label(footer, text="Open a scan to begin.",
                                style="Status.TLabel", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.file_label = ttk.Label(footer,
                                    text="Misaki %s - no image loaded"
                                    % APP_VERSION,
                                    style="Status.TLabel", anchor="e")
        self.file_label.pack(side="right")

    def _make_canvas(self, parent, column):
        # The canvas sits inside a holder that takes the space, and is centred
        # in it at the shape of what it is showing rather than stretched to
        # fill. A square section in a tall pane left a band of dead grey above
        # and below the picture, inside the pane's own border, which reads as
        # the pane being the wrong size rather than the picture being square.
        holder = ttk.Frame(parent)
        holder.grid(row=1, column=column, sticky="nsew",
                    padx=(0, 5) if column == 0 else (5, 0))
        canvas = tk.Canvas(holder, bg=THEME["canvas"], highlightthickness=1,
                           highlightbackground=THEME["pane_edge"],
                           width=360, height=360)
        canvas.place(relx=0.5, y=0, anchor="n", width=360, height=360)
        holder.bind("<Configure>", lambda e: self._fit_panes())
        canvas.bind("<Configure>", lambda e: self._redraw())

        # Both panes show the same region, so either can be painted on.
        canvas.bind("<Button-1>", self._on_paint_start)
        canvas.bind("<B1-Motion>", self._on_paint_move)
        canvas.bind("<ButtonRelease-1>", self._on_paint_end)
        canvas.bind("<Motion>", self._on_hover)
        canvas.bind("<Leave>", lambda e: canvas.delete("cursor"))
        return canvas

    def _build_sidebar(self, parent):
        side = ttk.Frame(parent, padding=(SIDE_GAP, 0, 0, 0))
        side.grid(row=0, column=1, sticky="nsew")

        self.h_var = tk.DoubleVar(value=0.5)
        self.w_var = tk.DoubleVar(value=0.5)
        # Declared here rather than beside the tabs: Division is a handle on
        # Window size and is built further up the sidebar, so the store both
        # share has to exist before either of them.
        self.vars = {}
        self.brush_vars = {}

        # The heading sits in a row of exactly the height the pane header is
        # held at, so the navigator's top edge lands on the previews' top edge.
        # Matching the titles' own padding is not enough any more: those titles
        # are centred in a taller row, so copying their padding put the
        # navigator a few pixels high.
        head = ttk.Frame(side, height=PANE_HEADER)
        head.pack(fill="x")
        head.pack_propagate(False)
        self.nav_head = ttk.Label(head, text="Navigator",
                                  style="Heading.TLabel")
        self.nav_head.pack(anchor="w", expand=True)

        # A holder of fixed height with the canvas placed inside it at the
        # frame's own shape, for the reason the previews got the same
        # treatment: stretched to the sidebar's width, a 3:2 frame drew a
        # 225 px thumbnail in a 300 px canvas and the 75 px left over was grey,
        # inside the border, reading as part of the control.
        # Width as well as height: with the Loupe block pinned, this holder
        # is what asks for the sidebar's width, and without it the column
        # shrank to the mode row and clipped every hint wrapped to HINT_WRAP.
        nav_holder = ttk.Frame(side, width=SIDEBAR_WIDTH, height=NAV_HEIGHT)
        nav_holder.pack(fill="x")
        nav_holder.pack_propagate(False)
        self.overview_canvas = tk.Canvas(nav_holder, bg=THEME["canvas"],
                                         highlightthickness=1,
                                         highlightbackground=THEME["edge"])
        self.overview_canvas.place(x=0, y=0, width=SIDEBAR_WIDTH,
                                   height=NAV_HEIGHT)
        nav_holder.bind("<Configure>", lambda e: self._fit_overview())
        self.overview_canvas.bind("<Button-1>", self._on_overview_click)
        self.overview_canvas.bind("<B1-Motion>", self._on_overview_click)
        self.overview_canvas.bind("<Configure>", self._on_overview_resize)


        # The two answers to "how do I cover the whole frame" - all of it at
        # once, or all of it in turn - side by side and drawn alike, because
        # they are alternatives to each other and each excludes the other. Both
        # are buttons rather than ticks in a column: neither is another view
        # option, each changes what the preview window *is*. The tick stays
        # inside the button, since they still have an on and an off.
        # As wide as the thumbnail above it, not as the sidebar, so that
        # anything packed to its right edge lands on the thumbnail's right
        # edge - which is where the Loupe belongs. `_fit_overview` keeps the
        # two widths in step as the frame's shape decides the thumbnail's.
        self.mode_row = modes = ttk.Frame(side, width=SIDEBAR_WIDTH,
                                          height=SECTION_PANE + 4)
        modes.pack(anchor="w", pady=(8, 0))
        modes.pack_propagate(False)

        # Stacked rather than side by side, so Survey keeps the left edge the
        # thumbnail has and the pair take one column's width instead of two.
        buttons = ttk.Frame(modes)
        buttons.pack(side="left", anchor="n")
        self.survey_var = tk.BooleanVar(value=False)
        self._survey_index = 0
        self.survey_btn = ttk.Checkbutton(buttons, text="Survey",
                                          style="Big.TCheckbutton",
                                          variable=self.survey_var,
                                          command=self._on_survey_toggle)
        self.survey_btn.pack(fill="x")
        self.whole_var = tk.BooleanVar(value=False)
        self.whole_btn = ttk.Checkbutton(buttons, text="Wide",
                                         style="Big.TCheckbutton",
                                         variable=self.whole_var,
                                         command=self._on_whole_toggle)
        self.whole_btn.pack(fill="x", pady=(4, 0))

        # Window size, declared before anything builds a handle on it: the
        # Loupe's scale is built by hand below rather than through `_slider`,
        # because it is the one that runs vertically.
        self.vars["window"] = (tk.DoubleVar(value=float(DEFAULTS["window"])),
                               True, [])

        # The Loupe: the cell it is showing, with the slider that sizes the
        # view down its left side. Vertical because down is the direction that
        # means closer - dragging it down zooms in, which is why the scale runs
        # from the division at the top to 32 at the bottom.
        self.loupe_block = ttk.Frame(modes)
        self.loupe_block.pack(side="right", anchor="n")
        self.nav_scale = ttk.Scale(self.loupe_block, from_=WINDOW_MAX, to=32,
                                   orient="vertical", length=SECTION_PANE,
                                   variable=self.vars["window"][0],
                                   command=lambda _v:
                                       self._on_param_change("window"))
        self.nav_scale.pack(side="left", fill="y", padx=(0, 6))
        self.section_canvas = tk.Canvas(self.loupe_block, bg=THEME["canvas"],
                                        width=SECTION_PANE, height=SECTION_PANE,
                                        highlightthickness=1,
                                        highlightbackground=THEME["edge"])
        self.section_canvas.pack(side="left")
        self.section_canvas.bind("<Button-1>", self._on_section_click)
        self.section_canvas.bind("<B1-Motion>", self._on_section_click)
        self.section_canvas.bind("<Configure>", lambda e: self._draw_section())
        self.slider_parts.setdefault("window", []).append((self.nav_scale, None))

        # The cell size, between the mode buttons and the Loupe. Vertical too,
        # with + on top: up is fewer, bigger cells and down is more, smaller
        # ones, which is the same sense as the slider beside it.
        self.div_stack = ttk.Frame(modes)
        self.div_stack.pack(side="left", expand=True, anchor="n")
        self.div_up = ttk.Button(self.div_stack, text="+", width=2,
                                 command=lambda: self._step_division(1))
        self.div_up.pack()
        self.div_label = ttk.Label(self.div_stack, width=5, anchor="center",
                                   font=FONT_BOLD,
                                   foreground=THEME["accent_dark"])
        self.div_label.pack(pady=(2, 2))
        self.div_down = ttk.Button(self.div_stack, text="−", width=2,
                                   command=lambda: self._step_division(-1))
        self.div_down.pack()

        # Everything below appears with the mode and leaves with it. A panel of
        # controls that only work in one mode is worse when it is present and
        # inert than when it is absent: inert controls still have to be read
        # before they can be dismissed.
        self.survey_tools = ttk.Frame(side)
        # Two sizes, because in survey mode there are two: how much of the frame
        # you have committed to inspecting, and how much of that you are looking
        # at. Division is the first, and the only new number here.
        #
        # Stepped rather than dragged, because its useful values are not a
        # continuum: what a division *means* is a grid, and most of the pixels
        # on a slider give the same grid as their neighbours. Each press moves
        # the frame by exactly one column, which is the smallest change there is
        # that changes anything - and + adds a column, so the button and the
        # cell count agree even though the size in the readout goes the other
        # way.
        # Registered by hand, since there is no scale to do it: the readout is
        # a plain label, but everything else - `params()`, `reset_params`,
        # `_sync_label` - reaches it through `vars` like any other setting.
        self.vars["division"] = (tk.DoubleVar(value=float(DEFAULTS["division"])),
                                 True, [self.div_label])
        self.survey_grid_note = ttk.Label(self.survey_tools, style="Hint.TLabel",
                                          wraplength=HINT_WRAP, text="")
        self.survey_grid_note.pack(anchor="w", padx=(2, 0), pady=(SLIDER_GAP, 0))

        # One line, and always the same line: what the Loupe is for is dragging
        # the view around inside a cell, and that is the only thing about it
        # worth a sentence.
        self.section_hint = ttk.Label(self.survey_tools, style="Hint.TLabel",
                                      wraplength=SIDEBAR_WIDTH, justify="left",
                                      text="Click or drag the Loupe to move "
                                           "the view.")
        self.section_hint.pack(anchor="w", padx=(2, 0), pady=(2, 0))

        # Bottom up, so the notebook takes whatever is left in between.
        footer = ttk.Frame(side)
        footer.pack(side="bottom", fill="x", pady=(8, 0))
        self.reset_btn = ttk.Button(footer, text="Reset parameters",
                                    command=self.reset_params)
        self.reset_btn.pack(side="left")
        # Not disabled with the rest of the controls: which key does what is a
        # question you can answer with no image open, and while a render runs.
        self.keys_btn = ttk.Button(footer, text="Keybinds", width=10,
                                   command=self.edit_keys)
        self.keys_btn.pack(side="right")

        # One row of view options above the footer: what the panes show on the
        # left, how many of them on the right.
        views = ttk.Frame(side)
        views.pack(side="bottom", fill="x", pady=(10, 0))

        self.mask_var = tk.BooleanVar(value=False)
        self.mask_btn = ttk.Checkbutton(views, text="Show mask",
                                        variable=self.mask_var,
                                        command=self._on_mask_toggle)
        self.mask_btn.pack(side="left")

        # Which pane the mask is drawn over. It used to be the right one, with
        # no say in it, which is the wrong default to hard-code: the mask is
        # easiest to judge against the untouched frame on the left, and easiest
        # to judge as a *repair* on the right. Both at once is allowed. They
        # name panes, which is why they are L and R rather than arrows - an
        # arrow beside a tick reads as "move", not "this one".
        self.mask_left = tk.BooleanVar(value=False)
        self.mask_right = tk.BooleanVar(value=True)
        self.mask_panes = ttk.Frame(views)
        for side_name, var in (("L", self.mask_left), ("R", self.mask_right)):
            ttk.Checkbutton(self.mask_panes, text=side_name, variable=var,
                            command=self._on_mask_toggle).pack(side="left",
                                                               padx=(0, 6))
        self.mask_panes.pack(side="left", padx=(10, 0))
        self.mask_panes.pack_forget()   # appears with the mask, not before it

        self.single_var = tk.BooleanVar(value=False)
        self.show_result = tk.BooleanVar(value=True)
        self.single_btn = ttk.Checkbutton(views, text="Single pane",
                                          variable=self.single_var,
                                          command=self._on_single_toggle)
        self.single_btn.pack(side="right")

        ttk.Label(side, text="Masks", style="Heading.TLabel").pack(
            side="top", anchor="w", pady=(10, 2))
        # Expanding, so spare height goes to the list rather than pooling as
        # a blank panel under the tab contents. A frame retouched properly
        # runs to dozens of masks, and four rows of a fifty-row list is a
        # scrollbar doing work a taller box would do for free.
        holder = ttk.Frame(side)
        holder.pack(side="top", fill="both", expand=True)
        self.mark_list = tk.Listbox(holder, height=4, activestyle="none",
                                    bg=THEME["raised"], fg=THEME["ink"],
                                    selectbackground=THEME["accent"],
                                    selectforeground="#ffffff", font=FONT_MONO,
                                    highlightthickness=1, borderwidth=0,
                                    highlightbackground=THEME["edge"],
                                    exportselection=False)
        self.mark_list.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(holder, orient="vertical",
                            command=self.mark_list.yview)
        self.mark_list.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.mark_list.bind("<<ListboxSelect>>", self._on_mark_select)

        # Directly under the thing they act on. In the Retouch tab they were a
        # scroll away from the list, and neither one refers to the tool you are
        # holding - they refer to the selected row.
        buttons = ttk.Frame(side)
        buttons.pack(side="top", fill="x", pady=(6, 0))
        self.new_btn = ttk.Button(buttons, text="New mask", width=10,
                                  command=self.new_mark)
        self.new_btn.pack(side="left")
        self.delete_btn = ttk.Button(buttons, text="Delete", width=8,
                                     command=self.delete_mark)
        self.delete_btn.pack(side="left", padx=(4, 0))
        self.clear_btn = ttk.Button(buttons, text="Clear all", width=9,
                                    command=self.clear_strokes)
        self.clear_btn.pack(side="left", padx=(4, 0))

        self.tabs = tabs = ttk.Notebook(side)
        tabs.pack(side="top", fill="both", expand=True, pady=(10, 0))
        self.detect_tab = detect_tab = ttk.Frame(tabs)
        retouch_tab = ttk.Frame(tabs)
        # Retouch first, and so selected when the window opens. It is the panel
        # you reach for on every frame - the global pass now starts off, and
        # what you do next is paint. Detect is where you go once, if the
        # automatic search is worth turning on for this scan.
        tabs.add(retouch_tab, text="  Retouch  ")
        tabs.add(detect_tab, text="  Detect  ")
        detect = self._scrolling_panel(detect_tab)
        retouch = self._scrolling_panel(retouch_tab)

        # Both tabs open with the same two rows in the same order - Fill, then
        # Window size - so a control that means the same thing on either side
        # does not move when you switch. What follows is what the tab is for.
        fill_row = ttk.Frame(detect)
        fill_row.pack(fill="x", pady=(0, 2))
        ttk.Label(fill_row, text="Fill", width=12).pack(side="left")
        self.fill_var = tk.StringVar(value=DEFAULT_FILL)
        self.fill_names = {"Texture": FILL_TEXTURE, "Smooth": FILL_BIHARMONIC,
                           "Median": FILL_MEDIAN, "Auto": FILL_AUTO}
        self.fill_labels = {v: k for k, v in self.fill_names.items()}
        self.fill_box = ttk.Combobox(fill_row, state="readonly", width=10,
                                     values=list(self.fill_names))
        self.fill_box.set(self.fill_labels[DEFAULT_FILL])
        self.fill_box.pack(side="left")
        self.fill_box.bind("<<ComboboxSelected>>", self._on_fill_pick)
        self.fill_note = ttk.Label(detect, style="Hint.TLabel",
                                   wraplength=HINT_WRAP,
                                   text=FILL_NOTES[DEFAULT_FILL])
        self.fill_note.pack(anchor="w", padx=(2, 0))

        row, _note = self._slider(detect, "window", "Window size", 32,
                                  WINDOW_MAX, True, "")
        self.window_rows.append((row, self.fill_note))

        self.auto_var = tk.BooleanVar(value=DEFAULT_AUTO)
        # Labelled experimental in the interface, not only in the README. Every
        # test this pass applies is morphological, so it cannot tell dust from
        # fine bright structure - foliage and aerials are claimed as dust, and
        # no slider setting separates the two. Retouching by hand is the
        # finished path; this is a rough first pass to be checked in Wide.
        ttk.Checkbutton(detect, text="Global detection (experimental)",
                        variable=self.auto_var,
                        command=self._on_auto_toggle).pack(anchor="w",
                                                           pady=(SLIDER_GAP, 0))
        ttk.Label(detect, style="Hint.TLabel", wraplength=HINT_WRAP,
                  text="Off: only your own masks are repaired. On, it searches "
                       "the whole frame - it misses about half the dust and can "
                       "claim foliage or wires, so check it in Wide before "
                       "saving."
                  ).pack(anchor="w", padx=(2, 0))

        self._slider(detect, "intensity", "Intensity", 1, 16, True,
                     "Radius the Median fill copies pixels from.")
        self._slider(detect, "sig_min", "Sigma min", 0, 10, False,
                     "Smallest speck to look for, roughly in pixels.")
        self._slider(detect, "sig_max", "Sigma max", 0, 10, False,
                     "Largest speck to look for, up to about %d px."
                     % (TOPHAT_RADIUS - 1))
        self._slider(detect, "threshold", "Threshold", 0, 100, False,
                     "How bright a speck must be, as % of white.")
        self._slider(detect, "spread", "Spread", 0.5, 8, False,
                     "How far a repair reaches past the speck.")

        self._build_retouch(retouch)
        self._bind_wheel()
        # The default fill does not read Intensity and there are no marks yet,
        # so Intensity
        # starts inert and should say so rather than waiting to be touched.
        self._refresh_intensity_state()
        # Same for the three sliders only the global pass reads: it starts off,
        # so they should open greyed rather than waiting for a first toggle.
        self._refresh_auto_state()
        self._refresh_survey()
        # Nothing is selected yet, so the per-mask controls should already look
        # it. Without this they opened live - dark labels on sliders that write
        # to a mask that does not exist - and only settled the first time a row
        # was clicked.
        self._on_mark_select()

    def _scrolling_panel(self, parent):
        """
        A tab body that scrolls once its contents outgrow the panel.

        Six sliders with an explanation apiece is taller than the sidebar at
        the minimum window size, and a notebook clips what does not fit rather
        than telling you about it - which is how Threshold and Spread came to
        be invisible. Maximised there is room to spare, so the bar shows only
        when there is somewhere to scroll to.
        """
        # An unsized Canvas asks for 378x265, which would make it - not the
        # navigator - the thing setting the sidebar's width. Ask for less than
        # it will get; pack expands it to fill the tab either way.
        # The requested height is a floor, not a size: pack hands out requests
        # before it shares surplus, so asking for enough to hold the paint
        # tools is what stops the mask list - which also expands - from
        # taking the room they need. 300 covers the tools and Brush size,
        # which end 275 px down the panel.
        canvas = tk.Canvas(parent, bg=THEME["bg"], highlightthickness=0,
                           width=200, height=300)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(10, 8, 6, 8))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def on_scroll(first, last):
            if float(first) <= 0.0 and float(last) >= 1.0:
                bar.pack_forget()
            else:
                bar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=on_scroll)
        canvas.pack(side="left", fill="both", expand=True)

        def resized(_event=None):
            # Give the frame the canvas's full height whenever the contents are
            # shorter than the panel. That leftover is what the spacers between
            # sliders expand into, so a tall window spreads the controls out
            # instead of stacking them at the top under a block of empty panel.
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Match the frame to the canvas so packing and wrapping behave as
            # they would in an ordinary panel.
            canvas.itemconfigure(window, width=canvas.winfo_width())

        inner.bind("<Configure>", resized)
        canvas.bind("<Configure>", resized)
        self._scrollers.append((canvas, inner))
        return inner

    def _bind_wheel(self):
        """
        Route the wheel to whichever scrolling panel is under the pointer.

        Tk delivers <MouseWheel> to the widget beneath the cursor, so a binding
        on the panel alone is dead over any slider or label inside it. Binding
        each descendant is the reliable way; it is done once, after the tabs
        are built.
        """
        for canvas, inner in self._scrollers:
            def wheel(event, canvas=canvas):
                first, last = canvas.yview()
                if first <= 0.0 and last >= 1.0:
                    return              # nothing to scroll; let the page be
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                return "break"

            stack = [canvas, inner]
            while stack:
                widget = stack.pop()
                widget.bind("<MouseWheel>", wheel)
                stack.extend(widget.winfo_children())

    def _build_retouch(self, parent):
        # The same two rows the Detect tab opens with, in the same order and at
        # the same height. Here Fill is the *selected mask's* method rather than
        # the global pass's, which is why it greys out until a mask is picked.
        fill_row = ttk.Frame(parent)
        fill_row.pack(fill="x", pady=(0, 2))
        ttk.Label(fill_row, text="Fill", width=12).pack(side="left")
        # Every mask names its own method. Inheriting from the Detect tab used
        # to be the default, and it meant changing one dropdown silently
        # re-rendered masks painted an hour earlier.
        self.mark_fill_box = ttk.Combobox(fill_row, state="readonly", width=10,
                                          values=list(self.fill_names))
        self.mark_fill_box.set(self.fill_labels[DEFAULT_FILL])
        self.mark_fill_box.pack(side="left")
        self.mark_fill_box.bind("<<ComboboxSelected>>", self._on_mark_fill_pick)
        self.mark_fill_note = ttk.Label(parent, style="Hint.TLabel",
                                        wraplength=HINT_WRAP, text="")
        self.mark_fill_note.pack(anchor="w", padx=(2, 0), pady=(2, 0))

        # The same control as the one in Detect, not a copy of it: retouching
        # is where you most want to widen the preview, and switching tabs to
        # reach it was the only way to do that.
        row, _note = self._slider(parent, "window", "Window size", 32,
                                  WINDOW_MAX, True, "")
        self.window_rows.append((row, self.mark_fill_note))

        ttk.Label(parent, text="Paint on either preview pane",
                  style="Heading.TLabel").pack(anchor="w", pady=(SLIDER_GAP, 0))

        self.tool_var = tk.StringVar(value="off")
        # One per row, each with its own line underneath. Across two columns
        # the labels were cramped and a single shared paragraph underneath had
        # to describe four tools at once, which made it long enough that nobody
        # would read it.
        for value, text, note in (
                ("off", "Off", ""),
                ("brush", "Sweep", "Drag over an artifact; searches inside it."),
                ("spot", "Spot", "Click a speck, or drag to size it."),
                ("protect", "Protect", ""),
                ("erase", "Eraser", "")):
            ttk.Radiobutton(parent, text=text, value=value,
                            variable=self.tool_var,
                            command=self._on_tool_change).pack(anchor="w",
                                                               pady=(4, 0))
            # A tool whose name says it all gets no line; an empty label would
            # still take a row of padding and leave a gap under it.
            #
            # These are indented under their radio button, so they wrap short by
            # that indent. Wrapping at the full width instead pushed the last
            # word of a two-line note off the panel, where there is no scrollbar
            # to reach it - a hint you cannot finish reading.
            if note:
                ttk.Label(parent, text=note, style="Hint.TLabel",
                          wraplength=HINT_WRAP - 18).pack(anchor="w",
                                                          padx=(18, 0))

        self._slider(parent, "size", "Brush size", 1, 64, True, "",
                     store=self.brush_vars)

        ttk.Label(parent, text="Selected mask", style="Heading.TLabel").pack(
            anchor="w", pady=(12, 0))
        self._slider(parent, "sensitivity", "Sensitivity", 0, 100, False, "",
                     store=self.brush_vars, command=self._apply_to_mark)
        self._slider(parent, "mark_spread", "Spread", 0.5, 8, False, "",
                     store=self.brush_vars, command=self._apply_to_mark)

        self.force_var = tk.BooleanVar(value=False)
        self.force_check = ttk.Checkbutton(parent, text="Repair whole stroke",
                                           variable=self.force_var,
                                           command=self._apply_to_mark)
        self.force_check.pack(anchor="w", pady=(6, 0))


        # Say what the recording is *for*. Left unexplained, a tick that writes
        # crops of your photographs to disk on every save is a reasonable thing
        # to be uneasy about. The answer is that the detector is morphological
        # and cannot be tuned past confusing dust with foliage; the way out is a
        # model trained on real corrections, and these are the only labels that
        # exist for one - especially the Protect strokes, which are confirmed
        # false positives and are otherwise almost impossible to collect.
        # The purpose goes in the label, not the hint. This sits at the bottom
        # of a scrolling panel with about one line of room below it, so a hint
        # long enough to explain itself is a hint cut off mid-sentence. The
        # label says what the crops are for; the README says why that is the
        # only way past a detector that cannot tell dust from foliage.
        self.record_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Record my corrections to train a detector",
                        variable=self.record_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(parent, style="Hint.TLabel", wraplength=HINT_WRAP - 18,
                  text="Saves crops to dataset/labels on this machine."
                  ).pack(anchor="w", padx=(18, 0))

    def _slider(self, parent, key, label, lo, hi, is_int, hint, store=None,
                command=None, width=12, gap=SLIDER_GAP):
        """
        One labelled slider. Calling this twice for the same key is allowed and
        gives a second handle on the *same* value: Window size belongs in both
        tabs, and a control that appears twice must never hold two answers.
        The variable is shared, so Tk moves both scales together, and both
        readouts are kept in the store so `_sync_label` updates each of them.

        Returns the row and its note, so a caller that needs to take a copy off
        screen again - as survey mode does with Window size - can put it back
        where it was.
        """
        store = self.vars if store is None else store
        source = BRUSH_DEFAULTS if store is self.brush_vars else DEFAULTS
        existing = store.get(key)

        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(gap, 0))

        var = existing[0] if existing else tk.DoubleVar(value=float(source[key]))
        text = ttk.Label(row, text=label, width=width)
        text.pack(side="left")
        value = ttk.Label(row, width=5, anchor="e", font=FONT_BOLD,
                          foreground=THEME["accent_dark"])
        value.pack(side="right")

        def moved(_v, k=key, s=store, extra=command):
            self._on_param_change(k, s)
            if extra is not None:
                extra()

        scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                          orient="horizontal", command=moved)
        scale.pack(side="left", fill="x", expand=True, padx=(4, 6))

        note = None
        if hint:
            note = ttk.Label(parent, text=hint, style="Hint.TLabel",
                             wraplength=HINT_WRAP)
            note.pack(anchor="w", padx=(2, 0), pady=(2, 0))

        readouts = (existing[2] if existing else []) + [value]
        store[key] = (var, is_int, readouts)
        # Kept so a slider can be greyed out when nothing reads it, each part
        # with the colour it wears when it is live. Extended rather than
        # replaced, or the second copy of a shared slider would leave the first
        # one behind, still active.
        parts = [(scale, None), (text, THEME["ink"]),
                 (value, THEME["accent_dark"])]
        if note is not None:
            parts.append((note, THEME["faint"]))
        self.slider_parts.setdefault(key, []).extend(parts)
        self._sync_label(key, store)
        return row, note

    def _set_slider_state(self, key, enabled):
        """
        Grey a slider that nothing is currently reading.

        Only the scale is really disabled - it is the part that has to stop
        accepting drags. The label, the readout and the note are dimmed by
        colour instead, because ttk's disabled state on a label draws a sunken
        box behind the text, and three of those stacked up read as a widget
        that has broken rather than one that is merely inactive.
        """
        for widget, normal in self.slider_parts.get(key, ()):
            if normal is None:
                widget.config(state="normal" if enabled else "disabled")
            else:
                widget.config(foreground=normal if enabled else THEME["dim"])

    # -- parameter plumbing -------------------------------------------------

    def _sync_label(self, key, store=None):
        store = self.vars if store is None else store
        var, is_int, readouts = store[key]
        text = str(int(round(var.get()))) if is_int else "%.1f" % var.get()
        if key in ("threshold", "sensitivity"):
            text += "%"
        for label in readouts:
            label.config(text=text)

    def params(self):
        out = {}
        for key, (var, is_int, _) in self.vars.items():
            out[key] = int(round(var.get())) if is_int else round(var.get(), 2)
        # difference_of_gaussians requires the high sigma to be the larger one.
        out["sig_max"] = max(out["sig_max"], out["sig_min"])
        out["intensity"] = max(1, out["intensity"])
        out["spread"] = max(0.1, out["spread"])
        out["fill"] = self.fill_var.get()
        out["auto"] = bool(self.auto_var.get())
        return out

    def brush_radius(self):
        return max(1, int(round(self.brush_vars["size"][0].get())))

    def _on_param_change(self, key, store=None):
        store = self.vars if store is None else store
        self._sync_label(key, store)
        # Window size and brush radius are the two controls that never reach
        # the result - neither appears in the render signature - so throwing
        # the render away for them would cost a re-render and buy nothing.
        if key not in VIEW_ONLY and key != "size":
            self._drop_render()

        if store is self.brush_vars:
            if key == "size":
                self._draw_cursor()
            else:
                self._schedule_preview()    # sensitivity changes the result
            return

        # Keep the two sigmas ordered so the pair always stays valid.
        if key in ("sig_min", "sig_max"):
            lo = self.vars["sig_min"][0].get()
            hi = self.vars["sig_max"][0].get()
            if key == "sig_min" and lo > hi:
                self.vars["sig_max"][0].set(lo)
                self._sync_label("sig_max")
            elif key == "sig_max" and hi < lo:
                self.vars["sig_min"][0].set(hi)
                self._sync_label("sig_min")
        # A mask keeps `spread: None` until you give it one of its own, and the
        # list prints the value it will actually use - so moving the global
        # slider changes what those rows say. Only rebuild when one of them is
        # actually inheriting.
        if key == "spread" and any(m.get("spread") is None for m in self.marks):
            self._refresh_marks()
        if (key in ("window", "division") and self.survey_var.get()
                and self.image is not None):
            # Division re-divides the frame, so the view is put back on a
            # numbered section rather than left off the grid the counter is
            # describing. Navigator only changes how much of the section is on
            # screen, so it keeps the view it has and re-clamps it.
            #
            # While Navigator is showing a whole section it goes on showing a
            # whole one as the division moves - resizing a section you are
            # looking at all of should not silently start hiding part of it.
            # Once you have zoomed, the zoom is yours and Division leaves it
            # alone except to clamp it.
            if (key == "division" and self._division_shown is not None
                    and int(round(self.vars["window"][0].get()))
                    >= self._division_shown):
                self._set_navigator(self.division())
            self._clamp_survey_sizes()
            if key == "division":
                y, x, h, w = window_origin(
                    self.image.shape, self.h_var.get(), self.w_var.get(),
                    self.params()["window"])
                self._go_to_section(
                    self._survey_home((y + h / 2.0, x + w / 2.0)))
            else:
                self._go_to_section(self._survey_index, keep_view=True)
            return
        self._draw_overview_rect()
        self._schedule_preview()

    def _on_fill_pick(self, _event=None):
        """Fill method changes the result, so the render goes with it."""
        self.fill_var.set(self.fill_names[self.fill_box.get()])
        self.fill_note.config(text=FILL_NOTES[self.fill_var.get()])
        self._refresh_intensity_state()
        self._drop_render()
        self._schedule_preview()

    def _refresh_intensity_state(self):
        """
        Grey out Intensity when nothing in the frame is median-filled.

        Intensity is the radius the median copies from, and the Smooth branch
        of `_fill_regions` never reads it: a Smooth group goes straight to the
        solver. Measured by sweeping the slider 1 to 16 on three scans, Smooth
        gave bit-identical output at every setting while Median and Auto moved
        by tens of thousands of levels. It does not reach detection either -
        `_segment` is not given it - and although it does widen the padding
        `_margin_for` asks for, that also measured zero difference under
        Smooth, because a speck's solve region sits well inside the window
        whichever padding it gets.

        It is not dead unconditionally, though, which is why this is a check
        rather than a deletion: fills are per mark, so one mask set to Median
        makes the slider live again even while the global pass is on Smooth.
        The condition below is exactly which groups `_segment` will build.
        """
        detect = self.fill_var.get()
        wanted = (FILL_MEDIAN, FILL_AUTO)
        live = bool(self.auto_var.get()) and detect in wanted
        for mark in self.marks:
            if (mark.get("fill") or detect) in wanted:
                live = True
                break
        self._set_slider_state("intensity", live)

    def selected_mark(self):
        """
        The mark the list is pointing at, or None for the Global row.

        Row 0 is the global pass. It is a row rather than a special case in
        another tab because everything that contributes to the mask should be
        visible and switchable in one place - and it stays listed when it is
        switched off, greyed, because a toggle whose own row disappears leaves
        nothing to click to bring it back.
        """
        picked = self.mark_list.curselection()
        if not picked or picked[0] == 0:
            return None
        index = picked[0] - 1
        return self.marks[index] if index < len(self.marks) else None

    def _refresh_marks(self, select=None):
        """Rebuild the list, keeping the selection where it can be kept."""
        keep = select
        if keep is None:
            current = self.selected_mark()
            keep = current["id"] if current else (
                0 if self.mark_list.curselection() else None)

        self.mark_list.delete(0, tk.END)
        self.mark_list.insert(tk.END, "Global detection   %s"
                              % ("on" if self.auto_var.get() else "off"))
        self.mark_list.itemconfig(
            0, foreground=THEME["ink"] if self.auto_var.get() else THEME["faint"])
        for mark in self.marks:
            kind = "Spot" if mark["kind"] == "spot" else "Sweep"
            # A forced mask ignores Sensitivity, so printing a percentage
            # there would be quoting a number that does nothing.
            sens = "forced" if mark.get("force") else "%.0f%%" % mark["sensitivity"]
            method = self.fill_labels.get(mark.get("fill") or self.fill_var.get())
            spread = mark.get("spread") or self.vars["spread"][0].get()
            self.mark_list.insert(tk.END, "Mask %-3d %-5s %-6s %-6s %.1f"
                                  % (mark["id"], kind, method, sens, spread))
        if keep == 0:
            self.mark_list.selection_set(0)
        else:
            for i, mark in enumerate(self.marks):
                if mark["id"] == keep:
                    self.mark_list.selection_set(i + 1)
                    self.mark_list.see(i + 1)
                    break
        self._on_mark_select()
        self._update_stroke_buttons()
        # A mark's own fill decides whether Intensity is read, so the list and
        # the slider's state go together: adding, deleting, editing a mark or
        # toggling the global pass all arrive here.
        self._refresh_intensity_state()

    def _on_mark_select(self, event=None):
        """
        Point the per-mark controls at whichever row is selected.

        `event` is set only when the listbox itself fired - a person clicked a
        row. `_refresh_marks` calls this with nothing, and that difference is
        load-bearing: selecting the Global row jumps to the Detect tab, and the
        list is rebuilt by a dozen things that are not a click. Moving the
        global Spread slider rebuilds it, so with the Global row selected the
        Retouch tab would throw you into Detect mid-drag, for no visible reason.
        """
        picked = self.mark_list.curselection()
        mark = self.selected_mark()
        live = mark is not None
        for key in ("sensitivity", "mark_spread"):
            self._set_slider_state(key, live)
        self.force_check.config(state="normal" if live else "disabled")
        # The fill box stays usable with nothing selected. With no mask it is
        # not describing one - it is the method the next mask you paint will be
        # created with, which is a choice you want to make before drawing
        # rather than after.
        self.mark_fill_box.config(state="readonly")
        self._refresh_mark_fill_note(mark)

        if mark is None:
            # Leave nothing behind from the last selection, or the box reads as
            # a setting for whatever is selected now.
            self._binding = True
            self.mark_fill_box.set(self.fill_labels[self._pending_fill])
            self._binding = False
            if picked and picked[0] == 0:
                # The global pass keeps its settings in the Detect tab, so
                # *clicking* it goes there rather than pretending otherwise.
                if event is not None:
                    self.tabs.select(self.detect_tab)
            self._draw_selection()
            return

        self._binding = True            # do not write back while loading
        self.brush_vars["sensitivity"][0].set(mark["sensitivity"])
        self._sync_label("sensitivity", self.brush_vars)
        self.brush_vars["mark_spread"][0].set(
            mark.get("spread") or self.vars["spread"][0].get())
        self._sync_label("mark_spread", self.brush_vars)
        self.force_var.set(bool(mark.get("force")))
        self.mark_fill_box.set(self.fill_labels.get(
            mark.get("fill"), self.fill_labels[DEFAULT_FILL]))
        self._binding = False
        # Only on a real click. Panning the preview because the list happened to
        # be rebuilt would move the frame under someone mid-edit.
        if event is not None:
            self._centre_on_mark(mark)
        self._draw_selection()

    def _refresh_mark_fill_note(self, mark):
        """Say what the fill method does, and whose it is."""
        if mark is None:
            self.mark_fill_note.config(
                text=NEW_MASK_PREFIX + FILL_NOTES[self._pending_fill])
            return
        self.mark_fill_note.config(
            text=FILL_NOTES[mark.get("fill") or self.fill_var.get()])

    def _on_mark_fill_pick(self, _event=None):
        """
        The fill box, which means two things depending on the selection.

        With a mask selected it edits that mask. With none, there is nothing to
        edit and it sets what the next one will be born with - so a method can
        be chosen before painting rather than only corrected afterwards.
        """
        if self._binding:
            return
        if self.selected_mark() is None:
            self._pending_fill = self.fill_names.get(self.mark_fill_box.get(),
                                                     DEFAULT_FILL)
            self._refresh_mark_fill_note(None)
            return
        self._apply_to_mark()

    def _apply_to_mark(self, *_args):
        """Write the per-mark controls back to the selected mark."""
        if self._binding:
            return
        mark = self.selected_mark()
        if mark is None:
            return
        mark["sensitivity"] = round(self.brush_vars["sensitivity"][0].get(), 1)
        mark["spread"] = round(self.brush_vars["mark_spread"][0].get(), 2)
        mark["force"] = bool(self.force_var.get())
        mark["fill"] = self.fill_names.get(self.mark_fill_box.get())
        self._edit_seq += 1
        self._drop_render()
        self._refresh_marks(select=mark["id"])
        self._schedule_preview()

    def _on_auto_toggle(self):
        """
        Grey out the sliders that only the automatic pass reads.

        Sigma and Threshold feed the global search and nothing else, so with it
        off they do nothing at all. Spread, Intensity and Window size still
        apply - the brush's own finds are expanded and filled the same way.
        """
        self._refresh_auto_state()
        self._drop_render()
        self._refresh_marks()
        self._schedule_preview()

    def _refresh_auto_state(self):
        """Grey the three sliders that only the global search reads."""
        live = bool(self.auto_var.get())
        for key in ("sig_min", "sig_max", "threshold"):
            self._set_slider_state(key, live)

    def _refresh_mask_panes(self):
        """L and R are on offer only when there are two panes to name."""
        wanted = bool(self.mask_var.get()) and not self.single_var.get()
        if wanted and not self.mask_panes.winfo_manager():
            self.mask_panes.pack(side="left", padx=(10, 0), after=self.mask_btn)
        elif not wanted and self.mask_panes.winfo_manager():
            self.mask_panes.pack_forget()
        # Both unticked with the mask on shows it nowhere, which reads as the
        # toggle being broken rather than as a choice not yet made.
        if wanted and not (self.mask_left.get() or self.mask_right.get()):
            self.mask_right.set(True)

    def _on_mask_toggle(self):
        """Show or hide the mask, and the pair of pane toggles with it."""
        self._refresh_mask_panes()
        self._update_titles()
        self._redraw()

    def _on_tool_change(self):
        painting = self.tool_var.get() != "off"
        cursor = "crosshair" if painting else ""
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.config(cursor=cursor)
            if not painting:
                canvas.delete("cursor")
        # The mask view is how you judge a stroke, so turn it on with the tool -
        # and in single pane make sure that is the side being shown, or you
        # would paint against the untouched original and see nothing happen.
        if painting:
            if self.single_var.get():
                self.show_result.set(True)
            if not self.mask_var.get():
                self.mask_var.set(True)
            self._on_mask_toggle()

    def _nudge_brush(self, delta):
        var = self.brush_vars["size"][0]
        var.set(max(1, min(64, var.get() + delta)))
        self._sync_label("size", self.brush_vars)
        self._draw_cursor()

    # -- retouching ---------------------------------------------------------

    def _to_canvas_xy(self, y, x):
        """Image coordinates back to a point on the preview canvas."""
        view = self._view
        oy_img, ox_img = view["origin"]
        ox, oy = view["offset"]
        return (ox + (x - ox_img) * view["scale"],
                oy + (y - oy_img) * view["scale"])

    def _spot_at_edge(self, point):
        """
        The spot whose rim the pointer is sitting on, if any.

        A spot keeps the centre and radius it was stamped from, not just the
        pixels, which is what makes it resizable after the fact - the circle can
        be drawn again at a new size instead of being scaled, so it stays a
        clean disc however many times it is adjusted.

        The grab distance is fixed in *screen* pixels. In image pixels it would
        shrink as you zoom out, until a small circle could not be caught at all.
        """
        if self._view is None:
            return None
        tol = max(2.0, SPOT_GRAB / self._view["scale"])
        best, best_gap = None, tol
        for mark in self.marks:
            if mark.get("centre") is None or not mark.get("radius"):
                continue
            cy, cx = mark["centre"]
            gap = abs(float(np.hypot(point[0] - cy, point[1] - cx))
                      - mark["radius"])
            if gap <= best_gap:
                best, best_gap = mark, gap
        return best

    def _merge_into(self, mark, canvas):
        """
        Fold a fresh stroke into an existing mask, growing its box to fit.

        Marks are stored as a small array plus an origin rather than a
        full-frame layer, so joining two is not an `or` - the union box has to
        be worked out and both pieces copied into it.
        """
        box = find_objects(canvas.astype(np.uint8))
        if not box or box[0] is None:
            return False                # the stroke landed off the frame
        sl = box[0]
        ny0, nx0, ny1, nx1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
        oy0, ox0 = mark["y"], mark["x"]
        oh, ow = mark["mask"].shape
        y0, x0 = min(oy0, ny0), min(ox0, nx0)
        y1, x1 = max(oy0 + oh, ny1), max(ox0 + ow, nx1)
        merged = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        merged[oy0 - y0:oy0 - y0 + oh, ox0 - x0:ox0 - x0 + ow] = mark["mask"]
        merged[ny0 - y0:ny1 - y0, nx0 - x0:nx1 - x0] |= canvas[sl]
        mark["mask"] = merged
        mark["y"], mark["x"] = y0, x0
        return True

    def _resize_spot(self, mark, radius):
        """Re-stamp a spot at a new radius, about the centre it was placed on."""
        if self.image is None or mark.get("centre") is None:
            return
        canvas = np.zeros(self.image.shape, dtype=bool)
        stamp(canvas, mark["centre"][0], mark["centre"][1], radius)
        box = find_objects(canvas.astype(np.uint8))
        if not box or box[0] is None:
            return                      # dragged off the frame entirely
        sl = box[0]
        mark["y"], mark["x"] = sl[0].start, sl[1].start
        mark["mask"] = canvas[sl].copy()
        mark["radius"] = radius

    def _to_image_xy(self, event):
        """Map a click on a preview canvas to image coordinates."""
        view = self._view
        if view is None or self.image is None:
            return None
        ox, oy = view["offset"]
        scale = view["scale"]
        cx = (event.x - ox) / scale
        cy = (event.y - oy) / scale
        vh, vw = view["size"]
        if not (0 <= cx < vw and 0 <= cy < vh):
            return None
        oy_img, ox_img = view["origin"]
        return int(oy_img + cy), int(ox_img + cx)

    def _on_paint_start(self, event):
        tool = self.tool_var.get()
        if tool == "off":
            # Nothing to paint with, so a click on the image compares instead.
            # In two-pane mode this is a no-op.
            self.toggle_side()
            return
        if self.image is None or self._saving:
            return
        point = self._to_image_xy(event)
        if point is None:
            return

        self._edit_seq += 1
        self._drop_render()

        if tool == "spot":
            # Pressing on an existing circle's rim resizes it rather than
            # starting a new one. Pressing anywhere else, inside it included,
            # still makes a new spot - so the only gesture that is taken away
            # is one nobody wants twice in the same place.
            grabbed = self._spot_at_edge(point)
            if grabbed is not None:
                cy, cx = grabbed["centre"]
                self._spot = {"mark": grabbed, "point": (cy, cx),
                              "screen": self._to_canvas_xy(cy, cx),
                              "widget": event.widget,
                              "radius": grabbed["radius"], "dragged": False}
                self._draw_spot_ring()
                return

        if tool in ("spot", "brush"):
            # A sweep joins the mask that is selected, so an artefact too awkward
            # to catch in one pass can be built up over several. Only sweeps:
            # a spot is one circle and keeps the centre and radius it was drawn
            # from, which a second circle in the same mask would have no answer
            # for. Use New mask to start a separate one.
            joining = self.selected_mark() if tool == "brush" else None
            if joining is not None and joining["kind"] != "brush":
                joining = None
            if joining is not None:
                self._stroke = {"into": joining, "kind": tool,
                                "radius": self.brush_radius(),
                                "canvas": np.zeros(self.image.shape, dtype=bool)}
                self._last_point = point
                self._apply_dab(point, point)
                self._schedule_preview()
                return

            # A new mark starts from the defaults, never from whichever mark is
            # currently selected. The Sensitivity slider and the force box are
            # bound to the selection, so reading them here made every mark
            # inherit the last one you looked at - select a spot, which is
            # forced by definition, and the next sweep silently became forced
            # too.
            self._stroke = {"id": self._next_mark, "kind": tool, "into": None,
                            "radius": self.brush_radius(),
                            "sensitivity": BRUSH_DEFAULTS["sensitivity"],
                            "spread": None,     # inherit until you change it
                            "fill": self._pending_fill,
                            "force": tool == "spot",
                            "canvas": np.zeros(self.image.shape, dtype=bool)}
            self._next_mark += 1
        else:
            self._stroke = {"kind": tool, "radius": self.brush_radius()}

        if tool == "spot":
            # A spot is sized by dragging out from where it was clicked, so
            # nothing is stamped until the button comes up. Stamping as the
            # drag went would leave every intermediate circle behind and
            # re-render the preview on each one.
            self._spot = {"mark": None, "point": point,
                          "screen": (event.x, event.y),
                          "widget": event.widget, "radius": self.brush_radius(),
                          "dragged": False}
            self._draw_spot_ring()
            return

        self._last_point = point
        self._apply_dab(point, point)
        self._schedule_preview()

    def _on_paint_move(self, event):
        if self._stroke is None and self._spot is None:
            return
        if self._spot is not None:
            point = self._to_image_xy(event)
            if point is not None:
                dy = point[0] - self._spot["point"][0]
                dx = point[1] - self._spot["point"][1]
                # The distance from the anchor, not the box the drag encloses,
                # so the mark stays a circle whichever direction you pull and
                # a diagonal drag does not quietly become an ellipse.
                self._spot["radius"] = max(1, int(round(float(np.hypot(dy, dx)))))
                self._spot["dragged"] = True
            # Outside the image `_to_image_xy` gives nothing; keep the last
            # size rather than snapping the circle to a pixel.
            self._draw_spot_ring()
            return
        point = self._to_image_xy(event)
        if point is None:
            return
        self._apply_dab(self._last_point, point)
        self._last_point = point
        self._draw_cursor(event)
        self._schedule_preview()

    def _on_paint_end(self, _event=None):
        """Close the stroke, and store a mark at its own size rather than the frame's."""
        stroke = self._stroke
        spot = self._spot
        self._stroke = None
        self._last_point = None
        self._spot = None
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.delete("spot")

        if spot is not None and spot["mark"] is not None:
            # Resizing an existing circle, not drawing a new one. A press that
            # never moved leaves it exactly as it was.
            if spot["dragged"]:
                self._resize_spot(spot["mark"], spot["radius"])
                self._refresh_marks(select=spot["mark"]["id"])
                self._schedule_preview()
            self._update_stroke_buttons()
            return

        if stroke is None:
            return
        if spot is not None:
            # A click that never moved keeps the brush size, so tapping a
            # speck still works exactly as it did before dragging existed.
            radius = spot["radius"] if spot["dragged"] else self.brush_radius()
            stroke["radius"] = radius
            # Kept so the circle can be drawn again later at another size,
            # rather than the stamped pixels being scaled.
            stroke["centre"] = spot["point"]
            stamp(stroke["canvas"], spot["point"][0], spot["point"][1], radius)
        if stroke["kind"] in ("spot", "brush"):
            canvas = stroke.pop("canvas")
            joining = stroke.pop("into", None)
            if joining is not None:
                if self._merge_into(joining, canvas):
                    self._refresh_marks(select=joining["id"])
                self._update_stroke_buttons()
                self._schedule_preview()
                return
            box = find_objects(canvas.astype(np.uint8))
            if box and box[0] is not None:
                sl = box[0]
                stroke["y"], stroke["x"] = sl[0].start, sl[1].start
                stroke["mask"] = canvas[sl].copy()
                self.marks.append(stroke)
                self._refresh_marks(select=stroke["id"])
        self._update_stroke_buttons()
        self._schedule_preview()

    def _apply_dab(self, start, end, stroke=None):
        """Rasterise one segment of a stroke."""
        stroke = self._stroke if stroke is None else stroke
        radius = stroke["radius"]

        if stroke["kind"] == "erase":
            # The Eraser deletes whole marks rather than nibbling holes in
            # them: a mark carries settings, and half a mark with settings
            # attached is a thing nobody asked for.
            hit = np.zeros(self.image.shape, dtype=bool)
            stamp_line(hit, start[0], start[1], end[0], end[1], radius, True)
            keep = []
            for mark in self.marks:
                mh, mw = mark["mask"].shape
                window = hit[mark["y"]:mark["y"] + mh, mark["x"]:mark["x"] + mw]
                if np.logical_and(window, mark["mask"]).any():
                    continue            # touched, so it goes
                keep.append(mark)
            if len(keep) != len(self.marks):
                self.marks = keep
                self._refresh_marks()
            stamp_line(self.protect_mask, start[0], start[1], end[0], end[1],
                       radius, False)
        elif stroke["kind"] == "protect":
            stamp_line(self.protect_mask, start[0], start[1], end[0], end[1],
                       radius, True)
        else:
            stamp_line(stroke["canvas"], start[0], start[1], end[0], end[1],
                       radius, True)

    def new_mark(self):
        """
        End the current mask, so the next stroke starts a fresh one.

        Sweeping used to make a new mask per stroke, which is wrong for the
        thing sweeping is for: a long hair is rarely caught in one pass, and
        three passes over one hair gave three masks with three sets of
        settings to keep in step. Strokes now join the selected mask, and this
        is how you say the next one is a different artefact.

        Nothing is created here. An empty mask in the list would be a row that
        claims nothing and repairs nothing; letting the selection go is the
        same statement without the debris.
        """
        if self.image is None or self._saving:
            return
        self.mark_list.selection_clear(0, tk.END)
        self._on_mark_select()

    def delete_mark(self):
        """Remove the selected mark."""
        mark = self.selected_mark()
        if mark is None or self._saving:
            return
        self.marks = [m for m in self.marks if m["id"] != mark["id"]]
        self._edit_seq += 1
        self._drop_render()
        self._refresh_marks()
        self._schedule_preview(immediate=True)

    def clear_strokes(self):
        if self.image is None or not (self.marks or self.protect_mask.any()):
            return
        self._edit_seq += 1
        self._drop_render()
        self.marks = []
        self.protect_mask[:] = False
        self._refresh_marks()
        self._update_stroke_buttons()
        self._schedule_preview(immediate=True)

    def _update_stroke_buttons(self):
        has = self.image is not None and bool(self.marks)
        self.new_btn.config(
            state="normal" if self.image is not None else "disabled")
        self.delete_btn.config(state="normal" if has else "disabled")
        self.clear_btn.config(
            state="normal" if (self.image is not None
                               and (self.marks
                                    or (self.protect_mask is not None
                                        and self.protect_mask.any())))
            else "disabled")
        # Marks are half of what decides whether Next is allowed.
        self._refresh_queue_state()

    def _on_hover(self, event):
        if self.tool_var.get() == "off":
            return
        # Over a spot's rim, show the circle you would resize instead of the
        # brush you would paint with, and change the pointer. Without that the
        # gesture is invisible - there is nothing to suggest the edge is a
        # handle until you happen to press on it.
        if self.tool_var.get() == "spot":
            point = self._to_image_xy(event)
            grabbed = self._spot_at_edge(point) if point else None
            event.widget.config(cursor="sizing" if grabbed else "crosshair")
            if grabbed is not None:
                for canvas in (self.before_canvas, self.after_canvas):
                    canvas.delete("cursor")
                cx_s, cy_s = self._to_canvas_xy(*grabbed["centre"])
                r = grabbed["radius"] * self._view["scale"]
                event.widget.create_oval(cx_s - r, cy_s - r, cx_s + r, cy_s + r,
                                         outline="#ffa726", width=2,
                                         tags="cursor")
                return
        self._draw_cursor(event)

    def _draw_selection(self):
        """
        Outline the selected mask on the preview, so the list and the picture
        agree about which one is being edited.

        Magenta, the same as the navigator's rectangle: both answer "where",
        and neither is part of the mask legend, whose red, green and blue mean
        something about the pixels rather than about the selection.
        """
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.delete("selected")
        mark = self.selected_mark()
        if mark is None or self._view is None or self.image is None:
            return
        mh, mw = mark["mask"].shape
        x0, y0 = self._to_canvas_xy(mark["y"], mark["x"])
        x1, y1 = self._to_canvas_xy(mark["y"] + mh, mark["x"] + mw)
        pad = 3
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.create_rectangle(x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                                    outline="#d94fb0", width=2, dash=(4, 3),
                                    tags="selected")

    def _centre_on_mark(self, mark):
        """Move the preview window onto the mask that was just picked."""
        if self.image is None or self.whole_var.get():
            return                      # Wide View is already showing it
        ih, iw = self.image.shape
        size = max(1, min(self.params()["window"], ih, iw))
        mh, mw = mark["mask"].shape
        if self.survey_var.get():
            # Land on a section, not between two, so the counter goes on
            # meaning something after a jump to a mask.
            self._go_to_section(self._section_at(mark["y"] + mh / 2.0,
                                                 mark["x"] + mw / 2.0))
            return
        top = mark["y"] + mh / 2.0 - size / 2.0
        left = mark["x"] + mw / 2.0 - size / 2.0
        span_y, span_x = ih - size, iw - size
        self.h_var.set(0.0 if span_y <= 0 else min(1.0, max(0.0, top / span_y)))
        self.w_var.set(0.0 if span_x <= 0 else min(1.0, max(0.0, left / span_x)))
        self._draw_overview_rect()
        self._schedule_preview()

    def _draw_spot_ring(self):
        """Show the circle a Spot will stamp, at the size the drag has reached."""
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.delete("cursor")
            canvas.delete("spot")
        if self._spot is None or self._view is None:
            return
        x, y = self._spot["screen"]
        r = self._spot["radius"] * self._view["scale"]
        self._spot["widget"].create_oval(x - r, y - r, x + r, y + r,
                                         outline="#ffa726", width=1, tags="spot")

    def _draw_cursor(self, event=None):
        """Outline the brush at the pointer, so its size is visible."""
        for canvas in (self.before_canvas, self.after_canvas):
            canvas.delete("cursor")
        if event is None or self._view is None or self.tool_var.get() == "off":
            return
        r = self.brush_radius() * self._view["scale"]
        colour = {"brush": "#ff3b30", "protect": "#4a9eff",
                  "spot": "#ffa726"}.get(self.tool_var.get(), "#ffffff")
        event.widget.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                                 outline=colour, width=1, tags="cursor")

    def reset_params(self):
        self._drop_render()
        self.fill_var.set(DEFAULT_FILL)
        # The dropdown is not bound to the variable, so it has to be put back
        # by hand - without this the box goes on reading "Median" while the
        # fill has already reverted to Smooth.
        self.fill_box.set(self.fill_labels[DEFAULT_FILL])
        self.fill_note.config(text=FILL_NOTES[DEFAULT_FILL])
        self._pending_fill = DEFAULT_FILL
        if self.selected_mark() is None:
            self._binding = True
            self.mark_fill_box.set(self.fill_labels[DEFAULT_FILL])
            self._binding = False
            self._refresh_mark_fill_note(None)
        for key, (var, _is_int, _label) in self.vars.items():
            var.set(float(DEFAULTS[key]))
            self._sync_label(key)
        # The global pass has a default like everything else here, and this is
        # the button that restores defaults. It used to be the one setting
        # Reset left alone, so a frame swept once stayed swept.
        self.auto_var.set(DEFAULT_AUTO)
        self._refresh_auto_state()
        self._refresh_marks()
        self._refresh_intensity_state()
        if self.survey_var.get() and self.image is not None:
            # Division has just gone back to its build-time value, which is not
            # a division of this frame - re-size the sections around the view.
            self._start_survey()
            return
        self._draw_overview_rect()
        self._schedule_preview()

    def _set_controls_state(self, state):
        self.reset_btn.config(state=state)
        self.single_btn.config(state=state)
        if state == "disabled":
            for button in (self.new_btn, self.delete_btn, self.clear_btn,
                           self.survey_prev, self.survey_next,
                           self.div_up, self.div_down):
                button.config(state="disabled")
        else:
            self._update_stroke_buttons()
            self._refresh_survey()
        self._refresh_render_state()

    # -- opening and saving -------------------------------------------------

    def open_image(self):
        if self._saving:
            return
        path = filedialog.askopenfilename(title="Open scan", filetypes=OPEN_TYPES)
        if path:
            self.queue, self.queue_pos = [], -1
            self.load_path(path)

    def open_folder(self):
        if self._saving:
            return
        folder = filedialog.askdirectory(title="Open a folder of scans")
        if folder:
            self._queue_folder(folder)

    def _queue_folder(self, folder):
        paths = queue_from_folder(folder)
        if not paths:
            messagebox.showinfo("Nothing to open",
                                "No scans found in %s." % folder)
            return
        self.queue, self.queue_pos = paths, -1
        self._queue_go(0)

    def _queue_go(self, pos):
        target = self.queue[pos]
        self.load_path(target)
        if self.path == target:         # load_path reports failure itself
            self.queue_pos = pos
        self._refresh_queue_state()

    def _unsaved(self):
        """
        Whether moving on would discard something.

        Work is a mark drawn or a render held. It counts as saved only while
        the marks are the ones that were saved and the render, if there is
        one, is the render that was written - so a mark added or a render
        redone after the save makes the scan unfinished again.
        """
        if not self.marks and self.render is None:
            return False
        if self._edit_seq != self._saved_seq:
            return True
        return self.render is not None and self.render_sig != self._saved_sig

    def queue_step(self, delta, skip=False):
        if not self.queue or self._saving or self._rendering:
            return
        pos = self.queue_pos + delta
        if not 0 <= pos < len(self.queue):
            return
        # Reachable by keyboard while the button is greyed.
        if not skip and self._unsaved():
            return
        self._queue_go(pos)

    def _refresh_queue_state(self):
        if not hasattr(self, "queue_bar"):
            return
        if not self.queue:
            self.queue_bar.pack_forget()
            return
        self.queue_bar.pack(side="right")
        self.queue_label.config(
            text="%d of %d" % (self.queue_pos + 1, len(self.queue)))
        busy = self.image is None or self._saving or self._rendering
        held = self._unsaved()
        first = self.queue_pos <= 0
        last = self.queue_pos >= len(self.queue) - 1
        self.queue_prev.config(
            state="disabled" if busy or held or first else "normal")
        self.queue_skip.config(state="disabled" if busy or last else "normal")
        self.queue_next.config(
            state="disabled" if busy or held or last else "normal")

    def load_path(self, path):
        """Load a scan from disk and show the first preview."""
        self.status.config(text="Loading %s..." % os.path.basename(path))
        self.update_idletasks()
        try:
            img = load_gray(path)
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            self.status.config(text="Failed to open %s" % os.path.basename(path))
            return

        # A portrait scan is turned on its side to work on, and turned back
        # to save. Everything in this window is laid out for a landscape
        # frame - two preview panes side by side, a wide navigator, a grid
        # of cells - and a portrait scan in it wastes half the width while
        # the thumbnail shrinks to a slot. Rotating is a view decision, so
        # it is undone before the pixels reach disk: `save_image` turns the
        # render back, and the file keeps the orientation it arrived with.
        self.source_shape = img.shape
        self.rotated = img.shape[0] > img.shape[1]
        if self.rotated:
            img = np.ascontiguousarray(np.rot90(img, -1))

        self.image = img
        self.path = path
        self.source_meta = read_source_meta(path)
        self.result = None
        self._edit_seq += 1
        self.render = None
        self.render_mask = None
        self.render_idle = None
        self.render_sig = None
        self._saved_seq = None
        self._saved_sig = None
        self.h_var.set(0.5)
        self.w_var.set(0.5)

        # Retouching belongs to one image; start clean.
        self.protect_mask = np.zeros(img.shape, dtype=bool)
        self.marks = []
        self._next_mark = 1
        self._refresh_marks()
        self._stroke = None
        self._spot = None
        self._update_stroke_buttons()

        # The file's own dimensions, not the working ones: what is on disk did
        # not change shape, and saying it did would be the label lying about
        # the thing it names.
        self.file_label.config(
            text="%s   -   %d x %d   -   %d-bit%s"
                 % (os.path.basename(path), self.source_shape[1],
                    self.source_shape[0], 8 * img.dtype.itemsize,
                    "   -   turned to edit" if self.rotated else ""))
        self._set_controls_state("normal")
        self._build_overview()
        # Sections are sized to the frame, so a new frame is a new grid - and
        # a survey of it starts at the beginning rather than wherever the last
        # scan had been stepped to.
        self._survey_index = 0
        self._surveyed = False
        if self.survey_var.get():
            self._start_survey()
        self._refresh_survey()
        self._schedule_preview(immediate=True)

    def save_image(self):
        if self.image is None or self._saving or self._rendering:
            return
        if not self.render_valid():
            # Reachable by keyboard even while the button is greyed.
            messagebox.showinfo(
                "Render first",
                "Press Final Render before saving.\n\n"
                "It processes the whole scan so you can check the result, and "
                "the save then writes exactly those pixels.")
            return

        stem = os.path.splitext(os.path.basename(self.path))[0]
        folder = os.path.dirname(self.path)

        # Default a 16-bit scan to TIFF, so its depth is not silently dropped.
        deep = self.image.dtype == np.uint16
        ext = ".tif" if deep else ".jpg"
        types = [("TIFF (16-bit)", "*.tif"), ("PNG (16-bit)", "*.png"),
                 ("JPEG (8-bit only)", "*.jpg")]
        if not deep:
            types = [("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")]

        if self.queue:
            # The dialog is the one-by-one cost a queue exists to remove: every
            # scan wants the name the dialog would have offered anyway, beside
            # its source. Its one job that still matters - asking before a
            # file is replaced - is kept.
            path = os.path.join(folder, "%s_dustfree%s" % (stem, ext))
            if os.path.exists(path) and not messagebox.askyesno(
                    "Replace file?",
                    "%s already exists.\n\nReplace it?"
                    % os.path.basename(path)):
                return
        else:
            path = filedialog.asksaveasfilename(
                title="Save cleaned scan",
                initialdir=folder,
                initialfile="%s_dustfree%s" % (stem, ext),
                defaultextension=ext,
                filetypes=types,
            )
            if not path:
                return

        self._saving = True
        self._set_controls_state("disabled")
        self.open_btn.config(state="disabled")
        self.status.config(text="Writing %s..." % os.path.basename(path))

        # The settings these pixels were rendered under, captured now. Labels
        # are written after the file lands, and the sliders stay live in the
        # meantime, so reading them then would record whatever the controls had
        # drifted to rather than what produced the result - and the settings
        # columns are the whole reason the manifest exists. Each mark is copied
        # too, since the per-mask controls can still write to one mid-save.
        self._label_snapshot = (self.params(), [dict(m) for m in self.marks])

        # The rendered array itself, not a fresh computation of it. This is the
        # point of the render step: what you inspected is what lands on disk,
        # rather than something recomputed afterwards that ought to agree.
        cleaned = self.render
        if self.rotated:
            # Back the way it came in. Two quarter turns in opposite
            # directions are exact - no resampling, no rounding - so the
            # pixels written are the pixels rendered.
            cleaned = np.ascontiguousarray(np.rot90(cleaned, 1))
        meta = self.source_meta

        def work():
            try:
                save_array(cleaned, path, meta)
            except Exception:
                self._post(self._save_done, path, traceback.format_exc())
            else:
                self._post(self._save_done, path, None)

        threading.Thread(target=work, daemon=True).start()

    def _save_done(self, path, error):
        self._saving = False
        self._set_controls_state("normal")
        self.open_btn.config(state="normal")
        snapshot, self._label_snapshot = self._label_snapshot, None
        if error:
            messagebox.showerror("Could not save image", error)
            self.status.config(text="Save failed.")
        else:
            downgraded = path.lower().endswith((".jpg", ".jpeg"))
            bits = 16 if (self.image.dtype == np.uint16 and not downgraded) else 8
            self.status.config(text="Saved %d-bit to %s" % (bits, path))
            self._saved_seq, self._saved_sig = self._edit_seq, self.render_sig
            self._refresh_queue_state()
            if snapshot is not None:
                self._export_labels(*snapshot)
        if self._close_when_done:
            self._close_when_done = False
            self._on_close()

    # -- recording corrections as labels -------------------------------------

    def _export_labels(self, params, marks):
        """
        Write out what the human decided, as labelled patches.

        `params` and `marks` are the snapshot taken when the save started, not
        the live controls: this runs after the write finishes, by which time
        the sliders may have moved.

        Two kinds come out of ordinary retouching, and they are worth very
        different amounts.

        A **Spot** is a positive: somebody looked at a speck and clicked on it.
        No detector was consulted, so nothing about it is circular - which is
        the whole reason to collect these rather than harvest dust with a
        filter. Harvesting was tried and it mostly returned sky beside antenna
        elements, because a morphological detector cannot help but reproduce
        its own confusions in the labels it generates.

        A **Sweep** stroke is a weaker positive, and worth keeping for a
        different reason. You localised it, but the extent came from the
        brush's own finer search, so it is not purely your judgement. What
        makes it valuable is `missed_by_default`: if you had to drop the
        threshold or raise sensitivity before the speck was caught, that column
        says how much of it the shipping settings still would not find. Dust
        the defaults miss is precisely what a learned detector would be for.
        Where a stroke found nothing at all it is dropped - painting over clean
        film should not become a label.

        A **Protect** stroke is a negative, and only where the detector
        actually fired: those pixels are a confirmed false positive - canopy,
        foliage, a roof edge. Nothing else produces that label. Protecting a
        region the detector never touched says nothing, so it is not recorded.

        Failure here must never cost the user their save, so everything is
        wrapped and reported rather than raised.
        """
        if not self.record_var.get() or self.image is None:
            return
        vetoes = self.protect_mask is not None and self.protect_mask.any()
        if not (marks or vetoes):
            return

        try:
            folder = data_path("dataset", "labels")
            os.makedirs(folder, exist_ok=True)
            stem = os.path.splitext(os.path.basename(self.path))[0]
            half = LABEL_PATCH // 2
            ih, iw = self.image.shape
            if ih < LABEL_PATCH or iw < LABEL_PATCH:
                return
            rows = []

            # Each mark is one label, carrying the settings it was made with -
            # which is now per-mark, so a label records the sensitivity that
            # actually produced it rather than whatever the slider was left on.
            # The slug that makes a tag unique. The patch origin alone does not:
            # it is clamped into the frame, so every mark within half a patch of
            # a corner lands on the same origin, and the second one written
            # would overwrite the first's pixels while both kept a manifest row
            # pointing at them. A mask's own id separates them, and stays put
            # across re-saves so saving twice replaces a label rather than
            # duplicating it.
            jobs = [(m["mask"], m["y"], m["x"],
                     "dust" if m["kind"] == "spot" else "dust-brushed", m,
                     "m%d" % m["id"])
                    for m in marks]
            if vetoes:
                spread_out, _count = label(self.protect_mask)
                for n, sl in enumerate(find_objects(spread_out), start=1):
                    if sl is not None:
                        jobs.append(((spread_out[sl] == n), sl[0].start,
                                     sl[1].start, "not-dust", None, "p%d" % n))

            for piece, my, mx, kind, mark, slug in jobs:
                mh, mw = piece.shape
                cy, cx = my + mh // 2, mx + mw // 2
                y0 = max(0, min(ih - LABEL_PATCH, cy - half))
                x0 = max(0, min(iw - LABEL_PATCH, cx - half))
                region = (slice(y0, y0 + LABEL_PATCH),
                          slice(x0, x0 + LABEL_PATCH))
                crop = self.image[region]

                truth = np.zeros((LABEL_PATCH, LABEL_PATCH), dtype=bool)
                ty0, tx0 = max(y0, my), max(x0, mx)
                ty1 = min(y0 + LABEL_PATCH, my + mh)
                tx1 = min(x0 + LABEL_PATCH, mx + mw)
                if ty0 >= ty1 or tx0 >= tx1:
                    continue
                truth[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = \
                    piece[ty0 - my:ty1 - my, tx0 - mx:tx1 - mx]

                # What the shipping defaults would find here unaided. The
                # difference against that is the whole point of recording: it
                # separates dust anyone could detect from dust that needed you.
                stock = dust_mask(crop, DEFAULTS["sig_min"], DEFAULTS["sig_max"],
                                  DEFAULTS["threshold"], DEFAULTS["spread"]) > 0.5
                missed = 0
                sens, forced = "", 0

                if kind == "dust-brushed":
                    # The stroke says where to look; the mark's own search says
                    # how far the speck actually extends.
                    sens, forced = mark["sensitivity"], int(bool(mark["force"]))
                    _soft, idle = dust_mask(
                        crop, params["sig_min"], params["sig_max"],
                        params["threshold"], params["spread"],
                        marks=[{"mask": truth, "sensitivity": mark["sensitivity"],
                                "force": mark["force"]}],
                        auto=params["auto"], want_found=True)
                    if idle is not None:
                        truth = np.logical_and(truth, ~idle)
                    if not truth.any():
                        continue            # swept, but nothing was there
                    missed = int(np.logical_and(truth, ~stock).sum())

                elif kind == "not-dust":
                    # Only the part the detector claimed is a label. The rest of
                    # the stroke is just somewhere you painted.
                    fired = dust_mask(crop, params["sig_min"], params["sig_max"],
                                      params["threshold"], params["spread"]) > 0.5
                    truth = np.logical_and(truth, fired)
                    if not truth.any():
                        continue

                else:                       # a spot: pure assertion
                    sens, forced = mark["sensitivity"], 1
                    missed = int(np.logical_and(truth, ~stock).sum())

                tag = "%s_%s_%05d_%05d_%s" % (stem, kind, y0, x0, slug)
                tifffile.imwrite(os.path.join(folder, tag + "_img.tif"),
                                 crop, compression="deflate")
                Image.fromarray((truth * 255).astype(np.uint8), "L").save(
                    os.path.join(folder, tag + "_mask.png"), optimize=True)
                rows.append({
                    "tag": tag, "source": os.path.basename(self.path),
                    "kind": kind, "y": y0, "x": x0,
                    "px": int(truth.sum()),
                    # How much of it the shipping defaults would miss. A label
                    # entirely missed is one the detector cannot reach at all.
                    "missed_by_default": missed,
                    # The settings that produced it, so the label can be
                    # reproduced and a change of defaults is detectable.
                    "sig_min": params["sig_min"], "sig_max": params["sig_max"],
                    "threshold": params["threshold"], "spread": params["spread"],
                    "sensitivity": sens, "force": forced,
                    "auto": int(bool(params["auto"]))})

            if not rows:
                return
            book = os.path.join(folder, "manifest.csv")
            fields = list(rows[0].keys())

            # Keyed by tag rather than appended, because the patch files are
            # written by tag: saving the same frame twice overwrites them, and
            # an appended row would then be a second entry describing pixels
            # that no longer exist. Rewriting the whole file keeps the manifest
            # and the folder saying the same thing.
            keep = {}
            if os.path.exists(book):
                with open(book, newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    # An older manifest with different columns cannot be merged
                    # without inventing values, so it is set aside intact.
                    if reader.fieldnames == fields:
                        keep = {r["tag"]: r for r in reader}
                    else:
                        os.replace(book, book[:-4] + "_old.csv")
            for row in rows:
                keep[row["tag"]] = row
            with open(book, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(keep.values())
            self.status.config(
                text="%s   -   recorded %d labelled patches to dataset/labels"
                     % (self.status.cget("text"), len(rows)))
        except Exception:
            # Never let bookkeeping break a save.
            sys.stderr.write("label export failed:\n" + traceback.format_exc())

    # -- final render -------------------------------------------------------

    def _render_signature(self):
        """
        Everything that changes the output, in one comparable value.

        Anything missing from this is a way for the cache to go stale without
        anyone noticing, which would be worse than not caching at all - a save
        would write pixels nobody looked at. Window size and Division are
        deliberately absent: they decide what you are shown, never what is
        computed for it.

        The brush layers are covered by `_edit_seq` rather than by hashing 12
        megapixels on every keystroke. It is bumped by every path that can
        touch them: painting, changing a mask's own settings, deleting a mask,
        clearing, and loading a scan.
        """
        params = self.params()
        for key in VIEW_ONLY:
            params.pop(key, None)
        return (self.path, self.image.shape, str(self.image.dtype),
                tuple(sorted(params.items())), self._edit_seq)

    def render_valid(self):
        return (self.render is not None
                and self.render_sig == self._render_signature())

    def render_full(self):
        """Process the whole frame once, for inspection and then for saving."""
        if self.image is None or self._saving or self._rendering:
            return
        self._rendering = True
        self._set_controls_state("disabled")
        self.open_btn.config(state="disabled")
        self.status.config(text="Rendering the whole scan, please wait...")

        sig = self._render_signature()
        params = self.params()
        image = self.image
        # Marks are stored at their own size, so they have to be laid back onto
        # the frame before a whole-image pass can use them. `process_window`
        # does this through the same helper for its padded region.
        marks = slice_marks(self.marks, 0, 0, image.shape[0], image.shape[1])
        protect = self.protect_mask

        def work():
            try:
                cleaned, mask, idle = remove_dust(
                    image,
                    marks=marks,
                    protect=protect,
                    want_found=True,
                    **algo_args(params),
                )
            except Exception:
                self._post(self._render_done, sig, None, None, None,
                           traceback.format_exc())
            else:
                self._post(self._render_done, sig, cleaned, mask, idle, None)

        threading.Thread(target=work, daemon=True).start()

    def _render_done(self, sig, cleaned, mask, idle, error):
        self._rendering = False
        self.open_btn.config(state="normal")
        if error:
            messagebox.showerror("Render failed", error)
            self.status.config(text="Render failed - see the message above.")
            self._set_controls_state("normal")
            return

        # Settings may have moved while it ran. Keeping a render that no longer
        # matches them is exactly the stale-cache case, so it is discarded.
        if sig != self._render_signature():
            self.status.config(
                text="Settings changed while rendering - render again.")
            self._set_controls_state("normal")
            return

        self.render = cleaned
        self.render_mask = mask
        self.render_idle = idle
        self.render_sig = sig
        self._set_controls_state("normal")
        self.status.config(
            text="Rendered the whole scan. Save writes exactly these pixels.")
        self._schedule_preview(immediate=True)

    def _drop_render(self):
        """Throw the render away. Called by anything that changes the output."""
        if self.render is None:
            return
        self.render = None
        self.render_mask = None
        self.render_idle = None
        self.render_sig = None
        self._refresh_render_state()

    def _refresh_render_state(self):
        """Save is only available while a matching render exists."""
        valid = self.render_valid()
        if self.image is None or self._saving or self._rendering:
            self.save_btn.config(state="disabled")
        else:
            self.save_btn.config(state="normal" if valid else "disabled")
        if hasattr(self, "render_btn"):
            self.render_btn.config(
                state="disabled" if (self.image is None or self._saving
                                     or self._rendering or valid) else "normal")
        self._refresh_queue_state()

    # -- navigator ----------------------------------------------------------

    def _on_overview_resize(self, _event=None):
        """Rebuild the thumbnail when the navigator changes size."""
        if self.image is None:
            return
        if self._overview_job is not None:
            self.after_cancel(self._overview_job)
        self._overview_job = self.after(80, self._build_overview)

    def _fit_overview(self):
        """
        Size the frame thumbnail to the frame's own shape.

        Left-aligned rather than centred: a portrait scan makes it narrow, and
        a narrow control hanging in the middle of the sidebar reads as adrift
        where one flush with the column above it reads as deliberate. Height is
        capped at NAV_HEIGHT so a portrait frame cannot push the sidebar down.
        """
        if self.image is None:
            return False
        holder = self.overview_canvas.master
        hw, hh = holder.winfo_width(), holder.winfo_height()
        if hw < 8 or hh < 8:
            return False
        ih, iw = self.image.shape
        aspect = iw / float(ih) if ih else 1.0
        width = min(hw, int(round(hh * aspect)))
        height = min(hh, int(round(hw / aspect)))
        canvas = self.overview_canvas
        moved = (abs(canvas.winfo_width() - width) > 1
                 or abs(canvas.winfo_height() - height) > 1)
        if moved:
            canvas.place_configure(width=width, height=height)
        # Whatever the thumbnail is, the mode row beneath it is the same
        # width - which is what puts the Loupe's right edge on the
        # thumbnail's right edge rather than on the sidebar's. Never narrower
        # than its contents, though: a portrait frame makes the thumbnail
        # 100 px, and the row would clip the controls rather than the
        # alignment.
        needed = sum(kid.winfo_reqwidth() for kid in
                     self.mode_row.winfo_children() if kid.winfo_manager())
        want = max(width, needed)
        if int(self.mode_row.cget("width")) != want:
            self.mode_row.configure(width=want)
        return moved

    def _build_overview(self):
        self._overview_job = None
        if self._fit_overview():
            return          # the <Configure> that follows rebuilds it to size
        canvas = self.overview_canvas
        # Before the window is mapped winfo_* reports 1, so fall back to the
        # configured size.
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = int(canvas["width"]), int(canvas["height"])

        # Through to_pil, not Image.fromarray: mode "L" is 8-bit, so handing it
        # a 16-bit array decodes the buffer wrongly, and the noise that injects
        # folds into visible rings once the frame is reduced 15x. Measured on
        # the antenna scan, that path carried 17% of its energy above the
        # thumbnail's Nyquist limit against 1.3% for this one.
        full = to_pil(self.image)
        thumb = fit(full, (cw - 2, ch - 2), shrink=Image.BOX)
        self.overview = thumb
        self.overview_scale = thumb.width / self.image.shape[1]
        self.overview_offset = ((cw - thumb.width) // 2, (ch - thumb.height) // 2)

        self._photos["overview"] = ImageTk.PhotoImage(thumb)
        canvas.delete("all")
        canvas.create_image(self.overview_offset[0], self.overview_offset[1],
                            anchor="nw", image=self._photos["overview"])
        self._draw_overview_rect()

    def _draw_overview_rect(self):
        if self.image is None or self.overview is None:
            return
        canvas = self.overview_canvas
        canvas.delete("rect")

        y, x, h, w = self._target_region()
        s = self.overview_scale
        ox, oy = self.overview_offset

        if self.survey_var.get() and not self.whole_var.get():
            # The seams, drawn as the bands where neighbouring sections really
            # do overlap.
            #
            # This used to be a lattice of equal cells, which is not where the
            # sections are. On a 4410 px frame the cell edges fell at 735,
            # 1470, 2205... while the sections start at 722, 1444, 2167 - the
            # line drifting further right of the true edge at every seam, up to
            # 64 px by the last. The window box therefore looked shifted left
            # of its cell, and the right-hand seam looked starved. The sections
            # were even all along, at 77 px of overlap each; the drawing was
            # not, so it is now drawn from the same origins the stepping uses.
            origins, _rows, _cols = self._survey_layout()
            ih, iw = self.image.shape
            size = self.division()
            xs = sorted({ox_ for _oy, ox_ in origins})
            ys = sorted({oy_ for oy_, _ox in origins})
            # Faint on purpose. These say "the seams are here", which is
            # background information next to where you are; at a heavier
            # stipple they read as a grille laid over the picture.
            for i in range(len(xs) - 1):
                canvas.create_rectangle(ox + xs[i + 1] * s, oy,
                                        ox + (xs[i] + size) * s, oy + ih * s,
                                        fill=SEAM_TINT, stipple="gray12",
                                        outline="", tags="rect")
            for i in range(len(ys) - 1):
                canvas.create_rectangle(ox, oy + ys[i + 1] * s,
                                        ox + iw * s, oy + (ys[i] + size) * s,
                                        fill=SEAM_TINT, stipple="gray12",
                                        outline="", tags="rect")
            # The section is the solid rectangle here, and the window inside it
            # is the section pane's business - except when it is smaller, when
            # a hairline says so rather than leaving the frame navigator
            # claiming you are looking at more than you are.
            box = self.section_box()
            if box is not None:
                sy, sx, side = box
                canvas.create_rectangle(ox + sx * s, oy + sy * s,
                                        ox + (sx + side) * s,
                                        oy + (sy + side) * s,
                                        outline="#d94fb0", width=2, tags="rect")
                if w < side or h < side:
                    canvas.create_rectangle(ox + x * s, oy + y * s,
                                            ox + (x + w) * s, oy + (y + h) * s,
                                            outline="#ffffff", width=1,
                                            tags="rect")
                return

        # The thumbnail is greyscale, so a saturated hue reads against any part
        # of it regardless of how light or dark that part is - which a luminance
        # contrast alone would not, over a frame that runs from sky to shadow.
        # Brighter than the accent on purpose, so it is not mistaken for chrome.
        canvas.create_rectangle(ox + x * s, oy + y * s,
                                ox + (x + w) * s, oy + (y + h) * s,
                                outline="#d94fb0", width=2, tags="rect")

    def _on_whole_toggle(self):
        # Both answer "how do I cover the whole frame", so only one of them can
        # be the answer at a time.
        if self.whole_var.get() and self.survey_var.get():
            self.survey_var.set(False)
        self._refresh_survey()
        self._resize_panes_now()    # shape changes now, not when it lands
        self._draw_overview_rect()
        self._schedule_preview()

    def _on_overview_click(self, event):
        if self.image is None or self.overview is None or self.whole_var.get():
            return          # Wide View already shows everything
        ox, oy = self.overview_offset
        size = self.params()["window"]
        ih, iw = self.image.shape
        size = max(1, min(size, ih, iw))

        px = (event.x - ox) / self.overview_scale
        py = (event.y - oy) / self.overview_scale
        if self.survey_var.get():
            # In survey mode the window is always a numbered section, so a
            # click picks one rather than landing between two - otherwise the
            # counter would go on naming a section you had already left.
            self._go_to_section(self._section_at(py, px))
            return

        # Convert the click to a window origin, then back to a 0..1 position.
        px -= size / 2
        py -= size / 2
        self.w_var.set(0.0 if iw == size else min(1.0, max(0.0, px / (iw - size))))
        self.h_var.set(0.0 if ih == size else min(1.0, max(0.0, py / (ih - size))))

        self._draw_overview_rect()
        self._schedule_preview()

    # -- keyboard shortcuts -------------------------------------------------

    def _key_commands(self):
        """What each shortcut name actually does."""
        return {
            "open": self.open_image,
            "open_folder": self.open_folder,
            "queue_prev": lambda: self.queue_step(-1),
            "queue_next": lambda: self.queue_step(1),
            "queue_skip": lambda: self.queue_step(1, skip=True),
            "render": self.render_full,
            "save": self.save_image,
            "delete_mark": self.delete_mark,
            "new_mark": self.new_mark,
            "brush_down": lambda: self._nudge_brush(-2),
            "brush_up": lambda: self._nudge_brush(2),
            "prev_section": lambda: self._step_survey(-1),
            "next_section": lambda: self._step_survey(1),
            "toggle_survey": self._toggle_survey_key,
            "toggle_mask": self._toggle_mask_key,
            "toggle_single": self._toggle_single_key,
        }

    def _apply_keys(self):
        """
        Bind the current shortcuts, dropping whatever was bound before.

        Rebound wholesale rather than one at a time: `unbind` clears every
        binding for a sequence, so remembering which sequences we own is the
        only way to leave a rebound key genuinely free afterwards.
        """
        for sequence in getattr(self, "_bound_keys", ()):
            self.unbind(sequence)
        self._bound_keys = []
        commands = self._key_commands()
        for name, sequence in self.keys.items():
            action = commands.get(name)
            if action is None or not sequence:
                continue
            self.bind(sequence, lambda _e, fn=action: fn())
            self._bound_keys.append(sequence)

    def _toggle_survey_key(self):
        self.survey_var.set(not self.survey_var.get())
        self._on_survey_toggle()

    def _toggle_mask_key(self):
        self.mask_var.set(not self.mask_var.get())
        self._on_mask_toggle()

    def _toggle_single_key(self):
        self.single_var.set(not self.single_var.get())
        self._on_single_toggle()

    def edit_keys(self):
        """A window for rebinding the shortcuts, one keypress at a time."""
        if getattr(self, "_keys_window", None) is not None:
            try:
                self._keys_window.lift()
                return
            except tk.TclError:
                pass                    # it was closed; fall through and rebuild

        top = tk.Toplevel(self)
        self._keys_window = top
        top.title("Keyboard shortcuts")
        top.configure(bg=THEME["bg"])
        top.transient(self)
        top.resizable(False, False)
        try:
            top.iconbitmap(resource_path("assets", "app.ico"))
        except Exception:
            pass

        body = ttk.Frame(top, padding=(14, 12))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Keyboard shortcuts",
                  style="Heading.TLabel").grid(row=0, column=0, columnspan=3,
                                               sticky="w")
        ttk.Label(body, style="Hint.TLabel", wraplength=380,
                  text="Click a shortcut, then press the key you want. "
                       "Saved next to the application, so it survives a restart."
                  ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 10))

        working = dict(self.keys)
        buttons = {}
        state = {"waiting": None}

        def redraw():
            for name, button in buttons.items():
                button.config(text=key_label(working.get(name)))
            if state["waiting"] is not None:
                buttons[state["waiting"]].config(text="press a key...")

        def stop_waiting():
            if state["waiting"] is not None:
                state["waiting"] = None
                redraw()

        def start_waiting(name):
            state["waiting"] = name
            note.config(text="Press a key for this action, or Escape to keep "
                             "the one it has.", foreground=THEME["faint"])
            redraw()

        def on_key(event):
            name = state["waiting"]
            if name is None:
                return None
            if event.keysym == "Escape":
                stop_waiting()
                note.config(text="Left as it was.", foreground=THEME["faint"])
                return "break"
            sequence = key_from_event(event)
            if sequence is None:
                note.config(text="%s cannot be used here." % event.keysym,
                            foreground=THEME["accent_dark"])
                return "break"
            clash = [other for other, value in working.items()
                     if value == sequence and other != name]
            if clash:
                label = dict((n, l) for n, l, _d in KEY_ACTIONS)[clash[0]]
                note.config(text="%s is already %s." % (key_label(sequence),
                                                        label),
                            foreground=THEME["accent_dark"])
                return "break"
            working[name] = sequence
            state["waiting"] = None
            note.config(text="", foreground=THEME["faint"])
            redraw()
            return "break"

        for row, (name, label, _default) in enumerate(KEY_ACTIONS, start=2):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w",
                                             pady=2)
            button = ttk.Button(body, width=18,
                                command=lambda n=name: start_waiting(n))
            button.grid(row=row, column=1, sticky="e", padx=(20, 0), pady=2)
            buttons[name] = button

        note = ttk.Label(body, style="Hint.TLabel", wraplength=380, text="")
        note.grid(row=len(KEY_ACTIONS) + 2, column=0, columnspan=3, sticky="w",
                  pady=(8, 0))

        actions = ttk.Frame(body)
        actions.grid(row=len(KEY_ACTIONS) + 3, column=0, columnspan=3,
                     sticky="ew", pady=(10, 0))

        def restore():
            stop_waiting()
            working.update({n: d for n, _l, d in KEY_ACTIONS})
            note.config(text="Back to the shipped shortcuts - not saved yet.",
                        foreground=THEME["faint"])
            redraw()

        def apply_and_close():
            stop_waiting()
            self.keys = dict(working)
            self._apply_keys()
            if not save_keys(self.keys):
                messagebox.showwarning(
                    "Shortcuts not saved",
                    "The shortcuts are in effect for this session, but could "
                    "not be written to %s." % data_path(KEYBIND_FILE),
                    parent=top)
            close()

        def close():
            self._keys_window = None
            top.destroy()

        ttk.Button(actions, text="Restore defaults",
                   command=restore).pack(side="left")
        ttk.Button(actions, text="Cancel", command=close).pack(side="right")
        ttk.Button(actions, text="Save",
                   command=apply_and_close).pack(side="right", padx=(0, 6))

        top.bind("<KeyPress>", on_key)
        top.protocol("WM_DELETE_WINDOW", close)
        redraw()
        top.update_idletasks()
        # Over the middle of the main window, not the middle of the screen: a
        # settings window that opens on another monitor is a settings window
        # you have to go and find.
        top.geometry("+%d+%d" % (
            self.winfo_rootx() + (self.winfo_width() - top.winfo_width()) // 2,
            self.winfo_rooty() + max(40, (self.winfo_height()
                                          - top.winfo_height()) // 3)))
        top.focus_set()

    # -- survey mode --------------------------------------------------------

    def division(self):
        """How big a section is, held to something the frame can hold."""
        want = int(round(self.vars["division"][0].get()))
        if self.image is None:
            return want
        return max(32, min(want, *self.image.shape))

    def _clamp_survey_sizes(self):
        """
        Navigator at or below Division, and Division inside the frame.

        Enforced here rather than by the scale's own range: Window size has
        three handles on one variable, and the two in the tabs run to
        WINDOW_MAX, so whichever of them answered last decided the value. A
        window larger than the section it is supposed to sit in is the one
        combination these two controls can hold that means nothing - it would
        straddle a seam, and the counter would be naming one of two sections
        arbitrarily.
        """
        if self.image is None:
            return
        limit = self.division()
        if int(round(self.vars["division"][0].get())) != limit:
            self.vars["division"][0].set(float(limit))
            self._sync_label("division")
        if int(round(self.vars["window"][0].get())) > limit:
            self.vars["window"][0].set(float(limit))
            self._sync_label("window")

    def _step_division(self, delta):
        """
        One column coarser or finer.

        Stepping by pixels would be stepping by nothing: a division of 687 and
        one of 700 divide a 4283 px frame into the same 7 columns. What the
        control is really for is choosing a grid, so the step is a grid. For
        `cols` columns the smallest division that fits is
        `overlap + (width - overlap) / cols`, which inverts the count solved in
        `_survey_count`, and `delta` moves `cols` by one.
        """
        if self.image is None or not self.survey_var.get():
            return
        ih, iw = self.image.shape
        _origins, _rows, cols = self._survey_layout()
        # + gives more cells. The number in the readout is a cell *size*, so
        # it goes down as the count goes up; the line under it is what says
        # which way the grid actually moved.
        want = max(1, cols + delta)
        span = max(1, iw - SURVEY_OVERLAP)
        size = int(math.ceil(SURVEY_OVERLAP + span / float(want)))
        size = max(SURVEY_MIN, min(size, WINDOW_MAX, ih, iw))
        if size == self.division():
            return                      # already at the end of the range
        self.vars["division"][0].set(float(size))
        self._on_param_change("division")

    def _survey_layout(self):
        """The sections of the frame at the current division."""
        if self.image is None:
            return [], 0, 0
        return survey_grid(self.image.shape, self.division())

    def section_box(self):
        """The current section as (y, x, size), or None outside survey mode."""
        if not self.survey_var.get() or self.image is None:
            return None
        origins, _rows, _cols = self._survey_layout()
        if not origins:
            return None
        index = min(self._survey_index, len(origins) - 1)
        y, x = origins[index]
        return y, x, self.division()

    def _on_survey_toggle(self):
        """Turn stepping on, sizing the sections to the frame, or off again."""
        if not self.survey_var.get():
            # Nothing about the view changes on the way out. The window keeps
            # its size and its position, so the same pixels stay on screen and
            # the only difference is which controls are on offer - and the
            # sidebar narrowing, which the panes take up at once rather than
            # in a second step.
            self._refresh_survey()
            self._resize_panes_now()
            self._draw_overview_rect()
            return
        if self.whole_var.get():
            self.whole_var.set(False)
        self._start_survey()

    def _start_survey(self):
        """
        Divide the frame for the mode and open on a whole section of it.

        Navigator starts at the division, so the mode begins by showing what a
        section *is*. How big a section should be is the first decision here
        and it cannot be judged through a window smaller than one; zooming in
        comes after that decision, not before it.

        The section chosen is still the one already being looked at rather than
        the first, so the view zooms out around where you were instead of
        jumping to the corner.
        """
        if self.image is None:
            self._refresh_survey()
            return

        # Where the view is now, read before the sizes move.
        y, x, h, w = window_origin(self.image.shape, self.h_var.get(),
                                   self.w_var.get(), self.params()["window"])
        centre = (y + h / 2.0, x + w / 2.0)
        first = not self._surveyed
        self._surveyed = True

        size = survey_size(self.image.shape)
        var, _is_int, _readouts = self.vars["division"]
        if int(round(var.get())) != size:
            var.set(float(size))
            self._sync_label("division")
        self._set_navigator(self.division())
        self._go_to_section(0 if first else self._survey_home(centre))

    def _survey_home(self, centre):
        """
        Where a re-made grid should land.

        The first pass over a frame starts at cell 1, because a survey is a
        list to work through and lists start at the top. Coming back to it -
        or re-dividing it - is not the start of a pass, it is the middle of
        one, so it lands on the cell holding the last mask painted. Failing
        that, on the cell nearest whatever was on screen.
        """
        if self.marks:
            mark = self.marks[-1]
            mh, mw = mark["mask"].shape
            return self._section_at(mark["y"] + mh / 2.0, mark["x"] + mw / 2.0)
        return self._section_at(*centre)

    def _set_navigator(self, size):
        """Move Window size without going back round `_on_param_change`."""
        var, _is_int, _readouts = self.vars["window"]
        if int(round(var.get())) != int(size):
            var.set(float(size))
            self._sync_label("window")

    def _go_to_section(self, index, keep_view=False):
        """
        Move the preview window onto one section and say which it is.

        `keep_view` leaves the window where it is if it already fits inside the
        section, which is what makes entering the mode a no-op on screen.
        Otherwise the window is centred in the section - with Navigator at the
        division, the common case, centred and flush are the same thing.
        """
        origins, _rows, _cols = self._survey_layout()
        if not origins:
            return
        index = max(0, min(int(index), len(origins) - 1))
        self._survey_index = index
        sy, sx = origins[index]
        size = self.division()
        window = max(1, min(self.params()["window"], *self.image.shape))
        slack = max(0, size - window)

        if keep_view:
            now_y, now_x, _h, _w = window_origin(
                self.image.shape, self.h_var.get(), self.w_var.get(),
                self.params()["window"])
            wy = min(max(now_y, sy), sy + slack)
            wx = min(max(now_x, sx), sx + slack)
        else:
            wy, wx = sy + slack // 2, sx + slack // 2
        self._set_window_origin(wy, wx)
        self._refresh_survey()
        self._draw_overview_rect()
        self._schedule_preview()

    def _set_window_origin(self, y, x):
        """Put the preview window at an absolute origin in the frame."""
        ih, iw = self.image.shape
        size = max(1, min(self.params()["window"], ih, iw))
        span_y, span_x = ih - size, iw - size
        self.h_var.set(0.0 if span_y <= 0
                       else min(1.0, max(0.0, y / float(span_y))))
        self.w_var.set(0.0 if span_x <= 0
                       else min(1.0, max(0.0, x / float(span_x))))

    def _step_survey(self, delta):
        """Next or previous section, wrapping so a pass has no dead end."""
        if (not self.survey_var.get() or self.image is None
                or self._rendering or self._saving):
            return
        origins, _rows, _cols = self._survey_layout()
        if origins:
            self._go_to_section((self._survey_index + delta) % len(origins))

    def _section_at(self, y, x):
        """The section whose centre is nearest a point in the frame."""
        origins, _rows, _cols = self._survey_layout()
        if not origins:
            return 0
        half = self.division() / 2.0
        return min(range(len(origins)),
                   key=lambda i: (origins[i][0] + half - y) ** 2
                                 + (origins[i][1] + half - x) ** 2)

    def _on_section_click(self, event):
        """Drag inside the section pane to move the window within the section."""
        box = self.section_box()
        if box is None or self._section_view is None:
            return
        sy, sx, size = box
        ox, oy, scale = self._section_view
        window = max(1, min(self.params()["window"], *self.image.shape))
        # Centre the window on the click, then hold it inside the section -
        # which is what Division bounds: the drag cannot leave the section it
        # belongs to, or stepping would no longer cover the frame.
        wy = sy + (event.y - oy) / scale - window / 2.0
        wx = sx + (event.x - ox) / scale - window / 2.0
        slack = max(0, size - window)
        self._set_window_origin(min(max(wy, sy), sy + slack),
                                min(max(wx, sx), sx + slack))
        self._draw_overview_rect()
        self._draw_section()
        self._schedule_preview()

    def _draw_section(self):
        """The section pane: this section, with the preview window on it."""
        canvas = self.section_canvas
        canvas.delete("all")
        self._section_view = None
        box = self.section_box()
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if box is None or cw < 5 or ch < 5:
            return
        sy, sx, size = box
        ih, iw = self.image.shape
        crop = self.image[sy:min(sy + size, ih), sx:min(sx + size, iw)]
        if crop.size == 0:
            return

        thumb = fit(to_pil(crop), (cw - 2, ch - 2), shrink=Image.BOX)
        self._photos["section"] = ImageTk.PhotoImage(thumb)
        ox, oy = (cw - thumb.width) // 2, (ch - thumb.height) // 2
        canvas.create_image(ox, oy, anchor="nw", image=self._photos["section"])
        scale = thumb.width / float(crop.shape[1])
        self._section_view = (ox, oy, scale)

        wy, wx, wh, ww = window_origin(self.image.shape, self.h_var.get(),
                                       self.w_var.get(),
                                       self.params()["window"])
        canvas.create_rectangle(ox + (wx - sx) * scale, oy + (wy - sy) * scale,
                                ox + (wx - sx + ww) * scale,
                                oy + (wy - sy + wh) * scale,
                                outline="#d94fb0", width=2)

    def _refresh_survey(self):
        """
        Show or hide the survey controls, and keep them telling the truth.

        Everything here appears with the mode and leaves with it, rather than
        sitting greyed: these are not settings that happen to be unavailable,
        they are the controls of a mode you are not in.
        """
        origins, rows, cols = self._survey_layout()
        live = bool(self.survey_var.get()) and bool(origins)
        # Window size is the Loupe under another name while surveying, and
        # means nothing at all in Wide, where the whole frame is the window.
        # Either way it is a control that would sit there answering a
        # question nobody is asking.
        wanted = not (live or self.whole_var.get())
        for row, after in self.window_rows:
            if not wanted and row.winfo_manager():
                row.pack_forget()
            elif wanted and not row.winfo_manager():
                row.pack(after=after, fill="x", pady=(SLIDER_GAP, 0))
        if not live:
            self.survey_bar.place_forget()
            self.survey_tools.pack_forget()
            self.loupe_block.pack_forget()
            self.div_stack.pack_forget()
            self.nav_head.config(text="Navigator")
            self.survey_label.config(text="")
            self.section_canvas.delete("all")
            self._section_view = None
            self._division_shown = None
            # Off the leash again, or a hidden scale would go on holding Window
            # size down to the last division it was given.
            if float(self.nav_scale.cget("from")) != float(WINDOW_MAX):
                self.nav_scale.config(from_=WINDOW_MAX)
            return

        # Centred on the previews rather than on the window, and vertically in
        # the header row that is held open for it whether it is there or not.
        self.survey_bar.place(in_=self.before_canvas.master.master, relx=0.5,
                              y=PANE_HEADER // 2, anchor="center")
        # `winfo_manager`, not `winfo_ismapped`: a window that has not been
        # shown yet reports nothing as mapped, so the pack would be repeated -
        # and repeating it with `after=` is how a widget ends up reordered.
        if not self.survey_tools.winfo_manager():
            self.survey_tools.pack(after=self.mode_row, anchor="w", fill="x")
        if not self.loupe_block.winfo_manager():
            self.loupe_block.pack(side="right", anchor="n")
            self.div_stack.pack(side="left", expand=True, anchor="n")
            self._fit_overview()    # the row's contents changed, so its width did
        # The thumbnail is showing the grid now, not just a position, so it
        # is named for what it shows.
        self.nav_head.config(text="Cells")
        state = "disabled" if (self._rendering or self._saving) else "normal"
        for widget in (self.survey_prev, self.survey_next,
                       self.div_up, self.div_down):
            widget.config(state=state)

        # Navigator cannot exceed Division. Reconfiguring `to` is what enforces
        # it: Tk pulls a variable that sits above a scale's range back inside,
        # so dragging Division down carries Navigator with it. Only written
        # when it differs, so the change it provokes settles in one pass.
        limit = max(32, min(int(round(self.vars["division"][0].get())),
                            *self.image.shape))
        # `from_`, not `to`: the scale runs downwards, so its maximum - the
        # whole cell - is at the top.
        if int(float(self.nav_scale.cget("from"))) != limit:
            self.nav_scale.config(from_=limit)

        # Division can shrink the grid under a stepped-past index, so it is
        # clamped here rather than only where it is set.
        self._survey_index = min(self._survey_index, len(origins) - 1)
        count = len(origins)
        self.survey_label.config(text="Cell %d of %d"
                                      % (self._survey_index + 1, count))
        size = self.division()
        self.div_label.config(text=str(size))
        self.survey_grid_note.config(text="%d x %d = %d cells"
                                          % (cols, rows, count))
        self._division_shown = size
        self._draw_section()

    # -- single pane and which pane shows what ------------------------------

    def _on_single_toggle(self):
        """
        Collapse the two preview panes into one, or restore them.

        Two panes side by side each get a quarter of the window, and a
        difference you have to carry across a gap is a difference you will
        miss. One pane at full width, alternating in place, is the comparison
        an eye is actually good at. Nothing is reprocessed either way - both
        images are already in `self.result`.
        """
        left, right = self.before_canvas.master, self.after_canvas.master
        if self.single_var.get():
            self.before_title.grid_remove()
            right.grid_remove()
            self.result_title.grid_configure(column=0, columnspan=2)
            left.grid_configure(columnspan=2, padx=0)
        else:
            left.grid_configure(columnspan=1, padx=(0, 5))
            right.grid()
            self.before_title.grid()
            self.result_title.grid_configure(column=1, columnspan=1)
        self._refresh_mask_panes()
        self._update_titles()
        self._redraw()

    def toggle_side(self, _event=None):
        """Flip a single pane between the original and the result."""
        if not self.single_var.get() or self.result is None:
            return
        self.show_result.set(not self.show_result.get())
        self._update_titles()
        self._redraw()

    def _masked_panes(self):
        """
        Whether the left and the right pane are drawing the mask.

        In single pane there is only one, so L and R have nothing to name and
        are not asked: the one view carries the mask, and clicking it flips
        between that and the original, which is what the single view has
        always done.
        """
        on = bool(self.mask_var.get())
        if self.single_var.get():
            return on, on
        return on and bool(self.mask_left.get()), on and bool(self.mask_right.get())

    def _update_titles(self):
        left_masked, right_masked = self._masked_panes()
        before = "Dust mask" if left_masked else "Original"
        after = "Dust mask" if right_masked else "Dust removed"
        self.before_title.config(text=before)
        if not self.single_var.get():
            self.result_title.config(text=after)
            return
        showing = after if self.show_result.get() else before
        self.result_title.config(text="%s    -    click to compare" % showing)

    # -- preview pipeline ---------------------------------------------------

    def _schedule_preview(self, immediate=False):
        if self.image is None:
            return
        if self._preview_job is not None:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(1 if immediate else 120, self._request_preview)

    def _request_preview(self):
        self._preview_job = None
        if self.image is None:
            return

        params = self.params()
        y, x, h, w = self._target_region()
        self._seq += 1

        if self.render_valid():
            # Nothing left to compute: crop the render. Panning becomes
            # instant, and every window is exact by construction rather than
            # by a padding argument - there is no padded edge to cut a mark on.
            region = (slice(y, y + h), slice(x, x + w))
            result = (self.image[region], self.render[region],
                      self.render_mask[region],
                      None if self.render_idle is None else self.render_idle[region],
                      self.protect_mask[region])
            self._preview_ready(self._seq, result, params, (y, x, h, w))
            return

        if self.whole_var.get():
            self.status.config(text="Processing the whole scan...")
        with self._req_lock:
            self._req = (self._seq, self.image, y, x, h, w, params,
                         list(self.marks), self.protect_mask)
            self._req_ready.set()

    def _worker_loop(self):
        while True:
            self._req_ready.wait()
            with self._req_lock:
                req = self._req
                self._req = None
                self._req_ready.clear()
            if req is None:
                continue

            seq, image, y, x, h, w, params, marks, protect = req
            try:
                result = process_window(image, y, x, h, w, params,
                                        marks=marks, protect=protect)
            except Exception:
                self._post(self._preview_failed, seq, traceback.format_exc())
            else:
                self._post(self._preview_ready, seq, result, params,
                           (y, x, h, w))

    def _post(self, fn, *args):
        """Queue a callback for the main thread. Safe to call from any thread."""
        self._results.put((fn, args))

    def _drain_results(self):
        """Run queued worker callbacks on the main thread."""
        try:
            while True:
                fn, args = self._results.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        finally:
            if not self._closing:
                self._drain_job = self.after(30, self._drain_results)

    def _preview_ready(self, seq, result, params, origin):
        if seq != self._seq:
            return                      # a newer request has already been sent
        self.result = result
        self.result_origin = origin
        mask, idle = result[2], result[3]
        covered = float((mask > 0.5).mean()) * 100

        note = ""
        if idle is not None and idle.any():
            # A painted area where nothing was found is the one case worth
            # calling out, since the fix is to raise sensitivity.
            note = "  -  painted area with nothing found: raise sensitivity"
        _y, _x, rh, rw = self.result_origin
        if self.whole_var.get():
            where = "Wide view %dx%d" % (rw, rh)
        elif self.survey_var.get():
            where = "Cell %d of %d, %dpx" % (
                self._survey_index + 1, len(self._survey_layout()[0]), rw)
        else:
            where = "Window %dpx" % rw
        stage = "rendered" if self.render_valid() else "preview"
        detect = ("intensity %d, sigma %.1f-%.1f, threshold %.1f%%, spread %.1f"
                  % (params["intensity"], params["sig_min"], params["sig_max"],
                     params["threshold"], params["spread"])
                  if params.get("auto", True)
                  else "brush only, intensity %d, spread %.1f"
                       % (params["intensity"], params["spread"]))
        self.status.config(
            text="%s (%d-bit, %s)  -  repairing %.2f%% of it  -  %s%s"
                 % (where, 8 * self.image.dtype.itemsize, stage, covered,
                    detect, note))
        self._redraw()

    def _preview_failed(self, seq, error):
        if seq != self._seq:
            return
        self.status.config(text="Preview failed - see console for details.")
        sys.stderr.write(error)

    def _resize_panes_now(self):
        """
        Take up a sidebar width change in the same beat as the click.

        `pack_forget` on the survey controls does not narrow the sidebar until
        Tk next processes geometry, so sizing the panes straight afterwards
        used the width the sidebar is leaving - and they resized again a moment
        later when it settled. Flushing idle tasks first makes it one step.
        """
        # Nothing is drawn while the layout is being flushed. `update_idletasks`
        # runs the pending <Configure> handlers, and those call `_redraw` -
        # so without this the half-finished layout gets painted on its way
        # past, which is the same wrong frame by another route.
        self._settling = True
        try:
            self.update_idletasks()
            self._fit_panes()
            self.update_idletasks()
        finally:
            self._settling = False
        self._redraw()

    def _target_region(self):
        """
        The region the controls are currently asking for, delivered or not.

        Read from the controls rather than remembered, so it cannot drift out
        of step with them - there is no second copy of the answer to keep in
        sync with `_request_preview`.
        """
        if self.image is None:
            return self.result_origin
        return window_origin(self.image.shape, self.h_var.get(),
                             self.w_var.get(), self.params()["window"],
                             whole=self.whole_var.get())

    def _fit_panes(self):
        """
        Size the preview panes to the shape of the region they are showing.

        The holders keep the space; the canvases take the largest rectangle of
        the region's aspect that fits. A square window therefore gets square
        panes, and a portrait scan in Wide gets tall ones - the panes re-expand
        on their own, because the aspect is read from the result rather than
        assumed.

        One size is worked out for both, from the smaller of the two holders,
        rather than each pane sizing itself from its own. The two holders can
        differ by a pixel when the column split is odd, and by more when one is
        measured before the layout has settled; either way the panes came out
        unequal, which is visible as a mismatch a comparison view cannot afford.

        Returns whether anything moved, which is what lets `_redraw` skip a
        pass: Tk has not laid the canvas out when `place_configure` returns, so
        anything drawn now would be fitted to the size the canvas is leaving.
        """
        panes = [self.before_canvas] if self.single_var.get() else [
            self.before_canvas, self.after_canvas]
        sizes = [(c.master.winfo_width(), c.master.winfo_height())
                 for c in panes]
        hw = min(w for w, _h in sizes)
        hh = min(h for _w, h in sizes)
        if hw < 8 or hh < 8:
            return False

        # The shape of the region that has been *asked for*, not the last one
        # delivered. Leaving a mode changes the sidebar's width the instant it
        # is clicked, while the preview behind it takes seconds - so sizing
        # from the delivered region resized the panes twice: once to the old
        # aspect at the new width, then again when the result landed. Square
        # panes stretching wide and then squashing back is the distortion.
        region = self._target_region()
        aspect = 1.0
        if region is not None:
            _y, _x, rh, rw = region
            aspect = rw / float(rh) if rh else 1.0
        width = min(hw, int(round(hh * aspect)))
        height = min(hh, int(round(hw / aspect)))

        moved = False
        for canvas in panes:
            if (abs(canvas.winfo_width() - width) > 1
                    or abs(canvas.winfo_height() - height) > 1):
                canvas.place_configure(width=width, height=height)
                moved = True
        return moved

    def _redraw(self):
        if self._settling:
            return              # mid-relayout; the caller draws when it ends
        if self.result is None:
            self._show_welcome()
            return
        if self._fit_panes():
            # The panes are a different shape now, and the <Configure> that
            # follows will call this again with geometry Tk has actually
            # applied. Drawing here would put one frame on screen fitted to the
            # size the panes are leaving - which is what a mode change looked
            # like: the picture briefly the wrong size in the wrong place.
            return
        original, cleaned, mask, idle, protect = self.result

        left_masked, right_masked = self._masked_panes()
        masked = None
        if left_masked or right_masked:
            layers = {"mask": mask > 0.5, "idle": idle, "protect": protect}
            masked = to_pil(original, layers)     # drawn once, shown once or twice
        left = masked if left_masked else to_pil(original)
        right = masked if right_masked else to_pil(cleaned)

        if self.single_var.get():
            self._show(self.before_canvas,
                       right if self.show_result.get() else left, "single")
        else:
            self._show(self.before_canvas, left, "before")
            self._show(self.after_canvas, right, "after")
        # `_show` clears the canvas, so the outline is drawn after it rather
        # than only when the selection changes.
        self._draw_selection()

    def _show(self, canvas, pil_image, key):
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 5 or ch < 5 or self.result_origin is None:
            return
        shown = fit(pil_image, (cw - 4, ch - 4))
        self._photos[key] = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=self._photos[key])

        # Record how the crop sits on the canvas, so painting can map back.
        # Both panes are laid out identically, so one record serves both.
        y, x, h, w = self.result_origin
        self._view = {
            "offset": ((cw - shown.width) / 2, (ch - shown.height) / 2),
            "scale": shown.width / w,
            "origin": (y, x),
            "size": (h, w),
        }

    def _on_close(self):
        # `save_array` runs on a daemon thread writing straight to the path the
        # user picked, so destroying the window mid-write ends the interpreter
        # and takes the thread with it - leaving a truncated file where their
        # scan was supposed to be. A save is seconds, so the close is deferred
        # rather than refused, and `_save_done` finishes the job.
        if self._saving:
            self._close_when_done = True
            self.status.config(text="Finishing the save before closing...")
            return
        self._closing = True
        if self._drain_job is not None:
            try:
                self.after_cancel(self._drain_job)
            except tk.TclError:
                pass
        self.destroy()


def main():
    # Crisp text on high-DPI Windows displays.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    app = DustRemovalApp()
    try:
        dpi = app.winfo_fpixels("1i")
        if dpi > 0:
            app.tk.call("tk", "scaling", dpi / 72.0)
    except tk.TclError:
        pass

    # Allow "Open with", or dropping a file onto the executable. Deferred so
    # the window is mapped before the navigator sizes itself.
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.after(50, app.load_path, sys.argv[1])
    elif len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        app.after(50, app._queue_folder, sys.argv[1])

    app.mainloop()


if __name__ == "__main__":
    main()
