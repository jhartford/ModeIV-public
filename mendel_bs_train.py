import data_generator
from deepiv.networks import TreatmentNetwork, ResponseNetwork, MultinomialTreatmentNetwork, build_mlp
from deepiv.networks import DeepIVMulti as DeepIV
from ensembles import ModeIV, MeanEnsemble
import pytorch_lightning as pl
import tqdm
import torch
import torch.utils.data
import numpy as np
import os
import statsmodels.api as sm
import time
import pathlib
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import KBinsDiscretizer
import argparse

def print_perf(name, oos_perf, filename, args):
    print(f"{args.seed},{name},{args.n_train},{oos_perf},{args.epochs},{args.n_valid},{args.beta}", file=open(filename, "a"))

def perf_mode(dataset, models, filename, args, name=''):
    oos_perf = dataset.evaluate(lambda x,z,t: models(t,x,z).detach().cpu().numpy())
    print(f"Modal{'' if len(name)==0 else '-'+name} - Out of sample performance evaluated against the true function: {oos_perf}.")
    print_perf(f"modal{'' if len(name)==0 else '-'+name}", oos_perf, filename, args)

def perf_deepiv(dataset, filename, args, valid=None, name='', path=None, device='cuda', bootstrap=None):
    if bootstrap is not None:
        name += f"-{bootstrap}"
    print(name)
    if os.path.exists(path / f'treatment-{name}.pt'):
        return None
    deepiv = DeepIV(dataset, device=device, treatment_net="mixture_density", weight_decay_treat=0., weight_decay_resp=0.0, 
                dropout_rate=args.dropout, valid_inst=valid, uid=args.uid+name, discretizer=None, 
                linear_reponse=args.linear_response)
    if os.path.exists(path / f'treatment-{name}.pt'):
        deepiv.treatment_net.load_state_dict(torch.load(path / f'treatment-{name}.pt')())
        deepiv.response_net.load_state_dict(torch.load(path / f'response-{name}.pt')())
        if '[' in name:
            return deepiv
    else:
        deepiv.fit(epochs_response=args.epochs, epochs_treatment=args.epochs, batch_size=args.batch_size, boot_index=bootstrap)
        torch.save(deepiv.treatment_net.state_dict, path / f'treatment-{name}.pt')
        torch.save(deepiv.response_net.state_dict, path / f'response-{name}.pt')
    oos_perf = dataset.evaluate(lambda x,z,t: deepiv.predict(t,x,z))
    print(f"DeepIV{'' if len(name)==0 else '-'+name} - Out of sample performance evaluated against the true function: {oos_perf}.")
    print_perf(f"deepiv{'' if len(name)==0 else '-'+name}", oos_perf, filename, args)
    return deepiv

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
    parser.add_argument("--bs_id", default=None, type=int)
    parser.add_argument("--n_valid", type=int, default=30)
    parser.add_argument("--model_id", type=int, default=None, 
                        help="Train only a single member of the ensemble so ensemble can be trained in parallel")
    parser.add_argument("--n_inst", type=int, default=30)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--linear_response", action='store_true')
    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs('results/', exist_ok=True)
    filename = "results/ModalIV"
    filename += '.csv'

    if len(args.uid) > 0:
        # we're running a cluster job - 
        # Stupid hack: sleep for some random amount of time between
        # 0 and 20 seconds to minimize collisions between job writing
        time.sleep(np.random.rand() * 20)

    if not os.path.exists(filename):
        print("seed,model,n,mse,epochs,n_valid,beta", file=open(filename, "a"))

    dataset = data_generator.Mendel(args.n_train, n_inst=args.n_inst, n_valid=args.n_valid, seed=args.seed, beta=args.beta, 
                                    hetrogenous=True, var_scale=args.var_scale)

    valid = np.arange(args.n_inst)[dataset.valid == 1]
    exp = 'hetrogenous-bootstrap'
    response = 'linear' if args.linear_response else 'nonlinear'
    path = pathlib.Path.cwd() / 'saved' / exp / str(args.n_train) / str(args.var_scale).replace('.', '_') / str(args.n_valid) / response / str(args.seed)
    os.makedirs(path, exist_ok=True)
    if args.model_id is None or args.model_id == 0:
        perf_deepiv(dataset, filename, args, valid, name='valid', path=path, device=device, bootstrap=args.bs_id)
        perf_deepiv(dataset, filename, args, np.arange(args.n_inst), name='all', path=path, device=device, bootstrap=args.bs_id)
    models = {}
    if args.model_id is None:
        for i in range(args.n_inst):
            models[i] = perf_deepiv(dataset, filename + '-individuals', args, [i], name=f'marginal-[{i}]', path=path, device=device, bootstrap=args.bs_id)
    else:
        print(f"Running model {args.model_id}")
        models[args.model_id] = perf_deepiv(dataset, filename + '-individuals', args, [args.model_id],
                                name=f'marginal-[{args.model_id}]', path=path, device=device, bootstrap=args.bs_id)

    if args.model_id is None:
        ens = MeanEnsemble(models)
        perf_mode(dataset, ens.predict, filename, args, name="Mean")
        for p in [0.2, 0.3, 0.4, 0.5]:
            ens = ModeIV(models, percent_valid=p)
            perf_mode(dataset, ens.predict, filename, args, name=str(p))

if __name__ == "__main__":
    main()
