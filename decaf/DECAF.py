from collections import OrderedDict
from typing import Any, List, Optional, Union

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import decaf.logger as log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nonlin(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softmax":
        return nn.Softmax(dim=-1)
    else:
        raise ValueError(f"Unknown nonlinearity {name}")


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, data: torch.Tensor) -> torch.Tensor:
        E = torch.linalg.matrix_exp(data)
        f = torch.trace(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (E,) = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply


class Generator_causal(nn.Module):
    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        h_dim: int,
        f_scale: float = 0.1,
        dag_seed: list = [],
        nonlin_out: Optional[List] = None,
    ) -> None:
        super().__init__()

        if nonlin_out is not None:
            out_dim = 0
            for act, length in nonlin_out:
                out_dim += length
            if out_dim != x_dim:
                raise RuntimeError("Invalid nonlin_out")

        self.x_dim = x_dim
        self.nonlin_out = nonlin_out

        def block(in_feat: int, out_feat: int, normalize: bool = False) -> list:
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.shared = nn.Sequential(*block(h_dim, h_dim), *block(h_dim, h_dim)).to(
            DEVICE
        )

        if len(dag_seed) > 0:
            M_init = torch.rand(x_dim, x_dim) * 0.0
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = torch.rand(x_dim, x_dim) * 0.0
            for pair in dag_seed:
                M_init[pair[0], pair[1]] = 1

            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init, requires_grad=False).to(
                DEVICE
            )
        else:
            M_init = torch.rand(x_dim, x_dim) * 0.2
            M_init[torch.eye(x_dim, dtype=bool)] = 0
            M_init = M_init.to(DEVICE)
            self.M = torch.nn.parameter.Parameter(M_init).to(DEVICE)

        self.fc_i = nn.ModuleList(
            [nn.Linear(x_dim + 1, h_dim) for i in range(self.x_dim)]
        )
        self.fc_f = nn.ModuleList([nn.Linear(h_dim, 1) for i in range(self.x_dim)])

        for layer in self.shared.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer.weight)
                layer.weight.data *= f_scale

        for i, layer in enumerate(self.fc_i):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale
            layer.weight.data[:, i] = 1e-16

        for i, layer in enumerate(self.fc_f):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.weight.data *= f_scale

    def sequential(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        gen_order: Union[list, dict, None] = None,
        biased_edges: dict = {},
    ) -> torch.Tensor:
        out = x.clone().detach()

        if gen_order is None:
            gen_order = list(range(self.x_dim))

        for i in gen_order:
            x_masked = out.clone() * self.M[:, i]
            x_masked[:, i] = 0.0
            if i in biased_edges:
                for j in biased_edges[i]:
                    x_j = x_masked[:, j]
                    perm = torch.randperm(len(x_j))
                    x_masked[:, j] = x_j[perm]
            out_i = self.fc_i[i](torch.cat([x_masked, z[:, i].unsqueeze(1)], axis=1))
            out_i = nn.ReLU()(out_i)
            out_i = self.shared(out_i)
            out_i = self.fc_f[i](out_i).squeeze()
            out[:, i] = out_i

        if self.nonlin_out is not None:
            split = 0
            for act_name, step in self.nonlin_out:
                activation = get_nonlin(act_name)
                out[..., split : split + step] = activation(
                    out[..., split : split + step]
                )

                split += step

            if split != out.shape[-1]:
                raise ValueError("Invalid activations")

        return out


class Discriminator(nn.Module):
    def __init__(self, x_dim: int, h_dim: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

        for layer in self.model.parameters():
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_normal_(layer)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.model(x_hat)


class DECAF(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dag_seed: list = [],
        h_dim: int = 200,
        lr: float = 1e-3,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 32,
        lambda_gp: float = 10,
        lambda_privacy: float = 1,
        eps: float = 1e-8,
        alpha: float = 1,
        rho: float = 1,
        weight_decay: float = 1e-2,
        grad_dag_loss: bool = False,
        l1_g: float = 0,
        l1_W: float = 1,
        nonlin_out: Optional[List] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.iterations_d = 0
        self.iterations_g = 0

        log.info(f"dag_seed {dag_seed}")

        self.x_dim = input_dim
        self.z_dim = self.x_dim

        log.info(
            f"Setting up network with x_dim = {self.x_dim}, z_dim = {self.z_dim}, h_dim = {h_dim}"
        )
        # networks
        self.generator = Generator_causal(
            z_dim=self.z_dim,
            x_dim=self.x_dim,
            h_dim=h_dim,
            dag_seed=dag_seed,
            nonlin_out=nonlin_out,
        ).to(DEVICE)
        self.discriminator = Discriminator(x_dim=self.x_dim, h_dim=h_dim).to(DEVICE)

        self.dag_seed = dag_seed

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.generator(x, z)

    def gradient_dag_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the gradient of the output wrt the input. This is a better way to compute the DAG loss,
        but fairly slow atm
        """
        x.requires_grad = True
        z.requires_grad = True
        gen_x = self.generator(x, z)
        dummy = torch.ones(x.size(0))
        dummy = dummy.type_as(x)

        W = torch.zeros(x.shape[1], x.shape[1])
        W = W.type_as(x)

        for i in range(x.shape[1]):
            gradients = torch.autograd.grad(
                outputs=gen_x[:, i],
                inputs=x,
                grad_outputs=dummy,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            W[i] = torch.sum(torch.abs(gradients), axis=0)

        h = trace_expm(W**2) - self.hparams.x_dim

        return 0.5 * self.hparams.rho * h * h + self.hparams.alpha * h

    def compute_gradient_penalty(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        alpha = alpha.expand(real_samples.size())
        alpha = alpha.type_as(real_samples)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(real_samples.size(0), 1)
        fake = fake.type_as(real_samples)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def privacy_loss(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        return -torch.mean(
            torch.sqrt(
                torch.mean((real_samples - fake_samples) ** 2, axis=1)
                + self.hparams.eps
            )
        )

    def get_W(self) -> torch.Tensor:
        return self.generator.M

    def dag_loss(self) -> torch.Tensor:
        W = self.get_W()
        h = trace_expm(W**2) - self.x_dim
        l1_loss = torch.norm(W, 1)
        return (
            0.5 * self.hparams.rho * h**2
            + self.hparams.alpha * h
            + self.hparams.l1_W * l1_loss
        )

    def sample_z(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.z_dim, device=DEVICE)

    @staticmethod
    def l1_reg(model: nn.Module) -> float:
        l1 = torch.tensor(0.0, requires_grad=True)
        for name, layer in model.named_parameters():
            if "weight" in name:
                l1 = l1 + layer.norm(p=1)
        return l1

    def gen_synthetic(self, x: torch.Tensor, biased_edges: dict = {}) -> torch.Tensor:
        self.generator = self.generator.to(DEVICE)
        x = x.to(DEVICE)
        gen_order = self.get_gen_order()
        return self.generator.sequential(
            x,
            self.sample_z(x.shape[0]).type_as(x),
            gen_order=gen_order,
            biased_edges=biased_edges,
        )

    def get_dag(self) -> np.ndarray:
        return np.round(self.get_W().detach().cpu().numpy(), 3)

    def get_gen_order(self) -> list:
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0
        G = nx.from_numpy_matrix(dense_dag, create_using=nx.DiGraph)
        gen_order = list(nx.algorithms.dag.topological_sort(G))
        return gen_order

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> OrderedDict:
        # sample noise
        z = self.sample_z(batch.shape[0])
        z = z.type_as(batch)
        generated_batch = self.generator.sequential(batch, z, self.get_gen_order())

        # train generator
        if optimizer_idx == 0:
            self.iterations_d += 1
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            real_loss = torch.mean(self.discriminator(batch))
            fake_loss = torch.mean(self.discriminator(generated_batch.detach()))

            # discriminator loss
            d_loss = fake_loss - real_loss

            # add the gradient penalty
            d_loss += self.hparams.lambda_gp * self.compute_gradient_penalty(
                batch, generated_batch
            )
            if torch.isnan(d_loss).sum() != 0:
                raise ValueError("NaN in the discr loss")

            return d_loss
        elif optimizer_idx == 1:
            # sanity check: keep track of G updates
            self.iterations_g += 1

            # adversarial loss (negative D fake loss)
            g_loss = -torch.mean(
                self.discriminator(generated_batch)
            )  # self.adversarial_loss(self.discriminator(self.generated_batch), valid)

            # add privacy loss of ADS-GAN
            g_loss += self.hparams.lambda_privacy * self.privacy_loss(
                batch, generated_batch
            )

            # add l1 regularization loss
            g_loss += self.hparams.l1_g * self.l1_reg(self.generator)

            if len(self.dag_seed) == 0:
                if self.hparams.grad_dag_loss:
                    g_loss += self.gradient_dag_loss(batch, z)
            if torch.isnan(g_loss).sum() != 0:
                raise ValueError("NaN in the gen loss")

            return g_loss
        else:
            raise ValueError("should not get here")

    def configure_optimizers(self) -> tuple:
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        weight_decay = self.hparams.weight_decay

        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2),
            weight_decay=weight_decay,
        )
        return [opt_d, opt_g], []
