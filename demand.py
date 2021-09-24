import data_generator
from deepiv.networks import DeepIVMulti as DeepIV
from ensembles import ModeIV, MeanEnsemble
import torch
import torch.utils.data
import numpy as np
import os
import pathlib
import time
import argparse

parser = argparse.ArgumentParser("Experiment runner")
parser.add_argument("-n", "--n_train", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--bias", type=float, default=1.)
parser.add_argument("--variance", type=float, default=1.)
parser.add_argument("--cuda", action='store_true')
parser.add_argument("--uid", default="")
parser.add_argument("--save_models", action='store_true')
parser.add_argument("--n_inst", type=int, default=3)
parser.add_argument("--n_valid", type=int, default=None)
parser.add_argument("--max_inst", type=int, default=None)
parser.add_argument("--rho", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()

device = 'cuda' if args.cuda else 'cpu'

args.max_inst = args.n_inst - 1 if args.max_inst is None else args.max_inst

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if len(args.uid) > 0:
    # we're running a cluster job - sleep for some random amount of time between
    # 0 and 20 seconds to minimize collisions between job writing
    s = np.random.rand() * 20
    print(f"sleeping for {s} secs")
    time.sleep(s)

n_valid = np.random.randint(2,args.n_inst) if args.n_valid is None else args.n_valid
dataset_all = data_generator.MultiDemand(args.n_train, args.n_inst, seed=args.seed, 
                                         ypcor=args.rho, n_valid=n_valid, 
                                         scale_exclusion=args.bias * 60.,
                                         scale_noise=args.variance)

corr = str(tuple(np.arange(args.n_inst)[dataset_all.valid])).replace(', ', '-')
print(corr, n_valid)

filename = 'results/rerun_bias_variance.csv'
if not os.path.exists(f'{filename}'):
    with open(filename, 'a') as f:
        print('seed,model,n_inst,n_valid,bias,variance,mse', file=f)
performace_string = f"{args.seed},"+"{model}"+f",{args.n_inst},{args.n_valid},{args.bias},{args.variance},"+"{mse}"

path = pathlib.Path.cwd() / 'saved' / 'demand-new' / str(args.bias) / str(args.variance) / str(args.n_valid) / str(args.seed)
os.makedirs(path, exist_ok=True)

# Train the ensemble
models = {}
for i in range(args.n_inst):
    models[i] = DeepIV(dataset_all, device='cuda' if torch.cuda.is_available() else 'cpu', weight_decay_treat=0.0001,
                       weight_decay_resp=0.001, valid_inst=[i], uid=args.uid)

    models[i].fit(epochs_response=args.epochs, epochs_treatment=args.epochs, batch_size=args.batch_size)
    perf = dataset_all.evaluate(lambda x,z,t: models[i].predict(t,x,z))
    print(performace_string.format(model=f'deepiv-[{i}]', mse=perf),file=open(filename, 'a'))
    print(performace_string.format(model=f'deepiv-[{i}]', mse=perf))
    if args.save_models:
        torch.save(models[i].treatment_net.state_dict, path / f'treatment-{i}.pt')
        torch.save(models[i].response_net.state_dict, path / f'response-{i}.pt')


for p_ in range(2,9):
    ens = ModeIV(models, k=p_)
    perf = dataset_all.evaluate(lambda x,z,t: ens.predict(t,x,z).detach().cpu().numpy())
    print(performace_string.format(model=f'Eps-{p_}-min', mse=perf),file=open(filename, 'a'))
    print(performace_string.format(model=f'Eps-{p_}-min', mse=perf))

# Baselines

# Mean
ens = MeanEnsemble(models)
perf = dataset_all.evaluate(lambda x,z,t: ens.predict(t,x,z).detach().cpu().numpy())
print(performace_string.format(model=f'ens-mean', mse=perf),file=open(filename, 'a'))
print(performace_string.format(model=f'ens-mean', mse=perf))


# Opt
opt = DeepIV(dataset_all, device='cuda' if torch.cuda.is_available() else 'cpu', weight_decay_treat=0.0001, 
       weight_decay_resp=0.001, valid_inst=np.arange(args.n_inst)[dataset_all.valid == 1], uid=args.uid)

opt.fit(epochs_response=args.epochs,
        epochs_treatment=args.epochs)
perf = dataset_all.evaluate(lambda x,z,t: opt.predict(t,x,z))

print(performace_string.format(model=f'deepiv-opt', mse=perf),file=open(filename, 'a'))
print(performace_string.format(model=f'deepiv-opt', mse=perf))

if args.save_models:
    torch.save(opt.treatment_net.state_dict, path / f'treatment-valid.pt')
    torch.save(opt.response_net.state_dict, path / f'response-valid.pt')

# All
opt = DeepIV(dataset_all, device='cuda' if torch.cuda.is_available() else 'cpu', weight_decay_treat=0.0001, 
       weight_decay_resp=0.001, valid_inst=np.arange(args.n_inst), uid=args.uid)
opt.fit(epochs_response=args.epochs,epochs_treatment=args.epochs)
perf = dataset_all.evaluate(lambda x,z,t: opt.predict(t,x,z))
print(performace_string.format(model=f'deepiv-all', mse=perf),file=open(filename, 'a'))
print(performace_string.format(model=f'deepiv-all', mse=perf))

if args.save_models:
    torch.save(opt.treatment_net.state_dict, path / f'treatment-all.pt')
    torch.save(opt.response_net.state_dict, path / f'response-all.pt')

