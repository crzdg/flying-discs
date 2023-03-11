![License](https://img.shields.io/github/license/crzdg/flying-discs)
![Last Commit](https://img.shields.io/github/last-commit/crzdg/flying-discs)
![Coverage](https://raw.githubusercontent.com/gist/crzdg/629d8687524d945066e3335e00aa7543/raw/coverage-badge.svg)
![Tests](https://raw.githubusercontent.com/gist/crzdg/5c06ab3ff426558fa98e978a39b76a55/raw/tests-badge.svg)
![PyPI](https://img.shields.io/pypi/pyversions/flying-discs)
![PyPI](https://img.shields.io/pypi/status/flying-discs)
![PyPI](https://img.shields.io/pypi/v/flying-discs)
![PyPI](https://img.shields.io/pypi/dm/flying-discs)

# 🥏 Flying Discs

This Python repository contains modules and algorithms that can calculate trajectories of flying sports discs. The repository includes non-physical and physical model based algorithms and parameterizations for calculating the trajectories.

### 🤔 Why create this project?

This project can be used by researchers, sports enthusiasts, and anyone interested in studying or improving the performance of flying sports discs. The open-source nature of this project also allows for contributions and enhancements by the community, further improving its capabilities and usefulness.

## 🔬 Included disc models

- [x] [V. Morrison, “The physics of frisbees”, Electronic Journal of Classical Mechanics and Relativity, vol. 8, no. 48, 2005](http://web.mit.edu/womens-ult/www/smite/frisbee_physics.pdf)

## 🚀 Get Started

#### Installation

```bash
pip3 install flying-discs
```

### 👩‍🏫 Example

Code examples and Jupyter notebooks can be found under [examples](examples/).

#### Morrison Example

```python
from flying_discs.morrison.disc_morrison_linear import DiscMorrisonLinear
from flying_discs.morrison.morrison_constants import DiscMorrisonUltrastar

x0 = 0
y0 = 0
z0 = 1

disc = DiscMorrisonLinear(DiscMorrisonUltrastar(), x0, y0, z0)

timescale = 0.033
angle_of_attack = 5
v0 = 14
direction = 0

trajectory = disc.calculate_trajectory(
    timescale,
    alpha=angle_of_attack,
    v0=v0,
    direction=direction,
)
```

### 📃 Documentation

Some doc-strings are already added. Documentation is a work-in-progress and will be updated on a time by time basis.

### 💃🕺 Contribution

I welcome everybody contributing to this project. Please read the [CONTRIBUTING.md](./CONTRIBUTING.md) for more information.
Feel free to open an issue on the project if you have any further questions.

## 💻 Development

The repository provides tools for development using [hatch](https://hatch.pypa.io/latest/).

All dependencies for the project also can be found in a `requirements`-file.

Install the development dependencies.

```bash
pip3 install -r requirements/dev.txt
```

or 

```bash
pip3 install "flying-discs[dev]"
```

#### Tools

To run all development tools, type checking, linting and tests `hatch` is required.

```bash
make all
```

#### Type checking

Type checking with `mypy`.

```bash
make mypy
```

#### Linting

Linting with `pylint`.

```bash
make lint
```

#### Tests

Run tests with `pytest`.

```bash
make test
```

#### Update dependencies

Update python requirements with `pip-compile`.

```bash
make update
```

## 🧾 License

This repository is licensed under the GNU Lesser General Public License v3.0. See the LICENSE file for more information.

Please note that while the LGPLv3 allows for the use of the code in proprietary software projects, any modifications to the code must also be released under the LGPLv3.
