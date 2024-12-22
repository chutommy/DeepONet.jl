import torch
import numpy as np
import pathlib

pt = pathlib.Path("pt")
npz = pathlib.Path("npz")
npz.mkdir(exist_ok=True)

darcy_train = torch.load(pt / "darcy_train_16.pt", weights_only=True)
darcy_test = torch.load(pt / "darcy_test_16.pt", weights_only=True)
np.savez_compressed(
    npz / "darcys_flow.npz",
    x_train=darcy_train["x"].numpy(),
    y_train=darcy_train["y"].numpy(),
    x_test=darcy_test["x"].numpy(),
    y_test=darcy_test["y"].numpy(),
)


burgers_train = torch.load(pt / "burgers_train_16.pt", weights_only=True)
burgers_test = torch.load(pt / "burgers_test_16.pt", weights_only=True)
np.savez_compressed(
    npz / "burgers_equation.npz",
    x_train=burgers_train["x"].numpy(),
    y_train=burgers_train["y"].numpy(),
    x_test=burgers_test["x"].numpy(),
    y_test=burgers_test["y"].numpy(),
)
