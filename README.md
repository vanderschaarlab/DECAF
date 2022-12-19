# DECAF (DEbiasing CAusal Fairness)
[![Tests](https://github.com/vanderschaarlab/DECAF/actions/workflows/test_decaf.yml/badge.svg)](https://github.com/vanderschaarlab/DECAF/actions/workflows/test_decaf.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/vanderschaarlab/DECAF/blob/main/LICENSE)

Code Author: Trent Kyono and Boris van Breugel

This repository contains the code used for the "DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks" paper(2021).

## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Tests
You can run the tests using
```bash
pip install -r requirements_dev.txt
pip install .
pytest -vsx
```

## Contents

- `decaf/DECAF.py` - Synthetic data generator class - DECAF.
- `tests/run_example.py` - Runs a nonlinear toy DAG example.  The dag structure is stored in the `dag_seed` variable.  The edge removal is stored in the `bias_dict` variable.  See example usage in this file.

## Examples

Base example on toy dag:
```bash
$ cd tests
$ python run_example.py
```

An example to run with a dataset size of 2000 for 300 epochs:
```bash
$ python run_example.py --datasize 2000 --epochs 300
```

## Citing
```
@inproceedings{kyono2021decaf,
	title        = {DECAF: Generating Fair Synthetic Data Using Causally-Aware Generative Networks},
	author       = {van Breugel, Boris and Kyono, Trent and Berrevoets, Jeroen and van der Schaar, Mihaela},
	year         = 2021,
	booktitle    = {Conference on Neural Information Processing Systems(NeurIPS) 2021}
}
