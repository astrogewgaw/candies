<div style="font-family:JetBrainsMono Nerd Font">
<div align="center">

![The `candies` logo.][logo]

<center>
<sup>

The font used in the above logo is [**Candyday**][candyday] by [**Billy Argel**][billyargel].

</sup>
</center>
<br/>

![License][license-badge]
![Version][version-badge]

![Python versions][pyversions-badge]
[![Interrogate][interrogate-badge]][interrogate]

![Stars][stars-badge]
![Downloads][dm-badge]
[![Issues][issues-badge]][issues]

[![Gitmoji][gitmoji-badge]][gitmoji]
[![Code style: black][black-badge]][black]

</div>
<div align="justify">

## Contents

- [Rationale](#rationale)
- [Features](#features)
- [Installation](#installation)
- [Quick Guide](#quick-guide)

## Rationale

The [**SPOTLIGHT**][spotlight] project is a multibeam, commensal survey for FRBs and pulsars soon to be undertaken at the [**GMRT**][gmrt]. It is estimated that it will produce almost 1 PB worth of data per day[^1]. It is obviously impossible for a mere human being, or a group of them, to sift through and classify all the resulting FRB candidates. Thus, we plan to use [**`FETCH`**][fetch], a ML/DL-based classifier. However, every classifier requires some features per candidate, for both training and classification. The features required by `FETCH` are the dedispersed dynamic spectrum, and the DM-$t$ array. While the `FETCH` authors recommend using utility scripts packaged with their associated library, [**`your`**][your], to generate them, these scripts proved to be too slow for the low frequency data obtained from the GMRT, despite using the GPU.

Thus, we decided to develop [**`candies`**][candies], our feature extraction library for FRB candidates. Primarily developed by [**me**][me] as a part of my (ongoing) PhD (and with the help of engineers from [**NVIDIA**][nvidia]), it hopes to be a faster variant of `your`'s scripts used for the same thing: `your_candmaker.py` and `your_h5plotter.py`.

## Features

`candies` also plans to introduce additional features, such as:

- [x] Zooming into the DM-$t$ plane.
- [ ] Dealing with band-limited bursts.
- [ ] Support for GMRT's shared memory based ring buffers.
- [x] Dealing with both [**`PRESTO`**][presto]'s and [**AstroAccelerate**][aa]'s outputs.
- [x] Slightly more fancy plots (via the [**`proplot`**][proplot] library) 😅 ?

Currently, we only support processing raw data stored in the `SIGPROC` filterbank format, but support for additional formats might be added if there is any interest for the same from the community at large.

## Installation

`candies` is on PyPI, and hence can be installed by simply running:

```bash
pip install candies
```

In case you are interested in the cutting edge, you can also git clone and install it directly via Github:

```bash
git clone https://github.com/astrogewgaw/candies
cd candies
pip install -e .
```

where the `-e` flag is for an *editable install*; that is, any changes to the source code will reflect directly in your installation. If you don't want that, just remove the flag, or you can go ahead and use the more direct method:

```bash
pip install git+https://github.com/astrogewgaw/candies
```

## Quick Guide

For help with how to use candies, just type and run `candies` or `candies --help`. To process a list of FRB candidates stored as either a CSV file, a `PRESTO` `*.singlepulse` file, or an AstroAccelerate `*.dat` file, just use the command:

```bash
candies make /path/to/candidate/list -f /path/to/rawdata/file
```

To see what additional options you can use, just type `candies make --help`. To plot one or several HDF5 files generated as output, just run:

```bash
candies plot /path/to/HDF5/files
```

Once more, type `candies plot --help` to see what options are available to you.

</div>

[^1]: This is roughly equivalent to India's per day internet traffic!

[gitmoji]: https://gitmoji.dev
[nvidia]: https://www.nvidia.com
[me]: https://github.com/astrogewgaw
[black]: https://github.com/psf/black
[billyargel]: http://www.billyargel.com
[gmrt]: http://www.gmrt.ncra.tifr.res.in
[fetch]: https://github.com/devanshkv/fetch
[spotlight]: https://spotlight.ncra.tifr.res.in
[presto]: https://github.com/scottransom/presto
[candyday]: https://www.dafont.com/candyday.font
[candies]: https://github.com/astrogewgaw/candies
[proplot]: https://github.com/proplot-dev/proplot
[your]: https://github.com/thepetabyteproject/your
[issues]: https://github.com/astrogewgaw/candies/issues
[interrogate]: https://github.com/econchick/interrogate
[aa]: https://github.com/AstroAccelerateOrg/astro-accelerate
[logo]: https://raw.githubusercontent.com/astrogewgaw/logos/main/rasters/candies.png
[dm-badge]: https://img.shields.io/pypi/dm/candies?style=for-the-badge
[version-badge]: https://img.shields.io/pypi/v/candies?style=for-the-badge
[wheel-badge]: https://img.shields.io/pypi/wheel/candies?style=for-the-badge
[forks-badge]: https://img.shields.io/github/forks/astrogewgaw/candies?style=for-the-badge
[stars-badge]: https://img.shields.io/github/stars/astrogewgaw/candies?style=for-the-badge
[pyversions-badge]: https://img.shields.io/pypi/pyversions/candies.svg?style=for-the-badge
[issues-badge]: https://img.shields.io/github/issues/astrogewgaw/candies?style=for-the-badge
[license-badge]: https://img.shields.io/github/license/astrogewgaw/candies?style=for-the-badge
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[gitmoji-badge]: https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=for-the-badge
[interrogate-badge]: https://raw.githubusercontent.com/astrogewgaw/candies/main/assets/interrogate.svg
