import data_generator
import numpy as np
import os
import time
import pathlib
import argparse


def save_data(dataset, *args, **kwargs):
    '''
    Save the data so that it can be run in R for checking baselines.
    '''
    tensors = {k: t.detach().cpu().numpy()
               for k, t in dataset.training_data.items()}
    print({k: t.shape for k, t in tensors.items()})
    for k, t in tensors.items():
        np.savetxt(f"{k}.csv", t, delimiter=",")

    seed = np.random.randint(1e9)
    x, z, t, y, g_true = dataset.datafunction(200_000, seed, test=True)
    t_min, t_max = np.percentile(tensors['treat'].flatten(), [2.5, 97.5])
    t = np.linspace(t_min, t_max, 200_000).reshape(-1, 1)
    y = g_true(x, z, t)
    np.savetxt(f"feat_test.csv", x, delimiter=",")
    np.savetxt(f"treat_test.csv", t, delimiter=",")
    np.savetxt(f"inst_test.csv", z, delimiter=",")
    np.savetxt(f"response_test.csv", y, delimiter=",")
    return None


def main():
    parser = argparse.ArgumentParser("Experiment runner")
    parser.add_argument("-n", "--n_train", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--var_scale", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--uid", default="")
    parser.add_argument("--n_valid", type=int, default=30)
    parser.add_argument("--model_id", type=int, default=None,
                        help="Train only a single member of the ensemble so ensemble can be trained in parallel")
    parser.add_argument("--n_inst", type=int, default=30)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--linear_response", action='store_true')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = data_generator.Mendel(args.n_train, n_inst=args.n_inst, n_valid=args.n_valid, seed=args.seed, beta=args.beta,
                                    hetrogenous=True, var_scale=args.var_scale)

    valid = np.arange(args.n_inst)[dataset.valid == 1]
    exp = 'hetrogenous'
    response = 'linear' if args.linear_response else 'nonlinear'
    path = pathlib.Path.cwd() / 'saved' / exp / str(args.n_train) / \
        str(args.var_scale).replace('.', '_') / \
        str(args.n_valid) / response / str(args.seed)

    if args.model_id is None or args.model_id == 0:
        save_data(dataset)


if __name__ == "__main__":
    main()
