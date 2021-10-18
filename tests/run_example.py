import argparse

import networkx as nx
import pytorch_lightning as pl
from utils import gen_data_nonlinear

from decaf import DECAF
from decaf.data import DataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--d_updates", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--rho", type=float, default=2)
    parser.add_argument("--l1_W", type=float, default=1e-4)
    parser.add_argument("--logfile", type=str, default="default_log.txt")
    parser.add_argument("--datasize", type=int, default=1000)
    args = parser.parse_args()

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
    data_tensor = dm.dataset.x

    # sample default hyperparameters
    x_dim = dm.dims[0]
    z_dim = x_dim  # noise dimension for generator input. For the causal system, this should be equal to x_dim
    lambda_privacy = 0  # privacy used for ADS-GAN, not sure if necessary for us tbh
    lambda_gp = 10  # gradient penalisation used in WGAN-GP
    l1_g = 0  # l1 reg on sum of all parameters in generator
    weight_decay = 1e-2  # used by AdamW to regularise all network weights. Similar to L2 but for momentum-based optimization
    p_gen = (
        -1
    )  # proportion of points to generate (instead of copy from input) # Has to be negative for sequential sampling!
    use_mask = True

    # causality settings
    grad_dag_loss = False

    number_of_gpus = 0

    # model initialisation and train
    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        d_updates=args.d_updates,
        alpha=args.alpha,
        rho=args.rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=args.l1_W,
        p_gen=p_gen,
        use_mask=use_mask,
    )
    trainer = pl.Trainer(
        gpus=number_of_gpus,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=1,
        profiler=False,
        callbacks=[],
    )
    trainer.fit(model, dm)
    synth_data = (
        model.gen_synthetic(
            data_tensor, gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
        .detach()
        .numpy()
    )
    print("Data generated successfully!")
