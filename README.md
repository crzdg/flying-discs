![License](https://img.shields.io/github/license/crzdg/flying-discs)
![Last Commit](https://img.shields.io/github/last-commit/crzdg/flying-discs)
![Coverage](https://raw.githubusercontent.com/gist/crzdg/629d8687524d945066e3335e00aa7543/raw/coverage-badge.svg)
![Tests](https://raw.githubusercontent.com/gist/crzdg/5c06ab3ff426558fa98e978a39b76a55/raw/tests-badge.svg)
![PyPI](https://img.shields.io/pypi/pyversions/flying-discs)
![PyPI](https://img.shields.io/pypi/status/flying-discs)
![PyPI](https://img.shields.io/pypi/v/flying-discs)

# 🥏 Flying Discs

This Python repository contains modules and algorithms that can calculate trajectories of flying sports discs. The repository includes non-physical and physical model based algorithms and parameterizations for calculating the trajectories.

### 🤔 Why create this project?

This project can be used by researchers, sports enthusiasts, and anyone interested in studying or improving the performance of flying sports discs. The open-source nature of this project also allows for contributions and enhancements by the community, further improving its capabilities and usefulness.

## 🔬 Included disc models

#### V. Morrison, “The physics of frisbees,” Mount Allison University Physics Department, vol. 1, 2005.

The [Morrison](src/flying_discs/morrison) model is a Euler approximation of [S. A. Hummel, Frisbee Flight Simulation and Throw Biomechanics. University of California, Davis, 2003.](https://books.google.ch/books?id=KQlA7DJ323MC)

##### Constants

- **G. Stilley and D. Carstens**, “Adaptation of the Frisbee flight principle to delivery of special ordnance,” in 2nd Atmospheric Flight Mechanics Conference, in Guidance, Navigation, and Control and Co-located Conferences. American Institute of Aeronautics and Astronautics, 1972. doi: 10.2514/6.1972-982.
- **K. Yasuda**, “Flight and aerodynamic characteristics of a flying disk,” Japanese Soc. Aero. Space Sci, vol. 47, no. 547, pp. 16–22, 1999.
- **S. A. Hummel**, Frisbee Flight Simulation and Throw Biomechanics. University of California, Davis, 2003.
- **V. Morrison**, “The physics of frisbees,” Mount Allison University Physics Department, vol. 1, 2005.
- **W. J. Crowther** and J. R. Potts, “Simulation of a spin-stabilised sports disc,” Sports Engineering, vol. 10, no. 1, pp. 3–21, 2007.
- **L. Hannah**, “Constraining Frisbee Tracking Methods Through Bayesian Analysis of Flying Disc Models,” 2017.




## 🚀 Get Started

#### Installation

```bash
pip3 install flying-discs
```

### 👩‍🏫 Example

Jupyter notebooks can be found under [notebooks](notebooks/).

#### Morrison Example

```python
from flying_discs.morrison.constants import MorrisonUltrastar
from flying_discs.morrison.coordinates import MorrisonPosition3D
from flying_discs.morrison.linear import MorrisonLinearCalculator

disc = MorrisonLinearCalculator(MorrisonUltrastar())

Z0 = 1
TIMESCALE = 0.033
ANGLE_OF_ATTACK = 5
V0 = 14
DIRECTION_ANGLE = 0

throw = disc.calculate_trajectory(
    MorrisonPosition3D(z=Z0),
    V0,
    ANGLE_OF_ATTACK,
    DIRECTION_ANGLE,
    TIMESCALE,
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
