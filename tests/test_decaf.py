from typing import Tuple

import networkx as nx
import pytorch_lightning as pl
from utils import gen_data_nonlinear

from decaf import DECAF, DataModule


def generate_baseline() -> Tuple[DataModule, list, dict]:
    # causal structure is in dag_seed
    dag_seed = [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 0],
        [3, 0],
        [3, 6],
        [3, 7],
        [6, 9],
        [0, 8],
        [0, 9],
    ]
    # edge removal dictionary
    bias_dict = {6: [3]}  # This removes the edge into 6 from 3.

    # DATA SETUP according to dag_seed
    G = nx.DiGraph(dag_seed)
    data = gen_data_nonlinear(G, SIZE=2000)
    dm = DataModule(data.values)

    return dm, dag_seed, bias_dict


def test_sanity_params() -> None:
    dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm,
        dag_seed=seed,
    )

    assert model.generator is not None
    assert model.discriminator is not None
    assert model.x_dim == dummy_dm.dims[0]
    assert model.z_dim == dummy_dm.dims[0]


def test_sanity_train() -> None:
    dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm,
        dag_seed=seed,
    )
    trainer = pl.Trainer(max_epochs=2, logger=False)

    trainer.fit(model, dummy_dm)
