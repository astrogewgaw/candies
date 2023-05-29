<div align="center" style="font-family: JetBrainsMono Nerd Font">
<h1><code>candies</code></h1>
<h4><i>Sweet, sweet, candy-dates!</i></h4>
</div>

<div align="justify">

## Installation

Not on PyPI for now, so:

```bash
git clone https://github.com/astrogewgaw/candies
cd candies
pip install -e .
```

## Usage

Here is a simple example:

```python
from candies.core import Candies

# Load all candidates. In case your machine has multiple
# GPUs, don't forget to specify the device ID. This is
# passed on later to CuPy.
candies = Candies.get("dm500.csv", device=1)

# Select the 24th candidate, create its DM v/s time plot,
# and save it to disk as an HDF5 file. Support for more
# formats on the way!
candy = candies[24]
candy.calcdmt()
candy.plot()
candy.save()
```

</div>
