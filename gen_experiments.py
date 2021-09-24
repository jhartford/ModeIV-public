import argparse
import re
import os
from itertools import product

header='''#!/bin/sh
# Script for running serial program, diffuse.
#SBATCH --time=03-00:00            # time (DD-HH:MM)
#SBATCH --mem=8000M
#SBATCH --array=1-{n_jobs}
source activate robust

cd {pwd}

case $SLURM_ARRAY_TASK_ID in'''

FOOTER = '''esac
'''

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

parser = argparse.ArgumentParser(description='Job builder script')
parser.add_argument('callstring')
parser.add_argument('--local', action='store_true', help="build a simple local job - removes all the slurm headers, etc.")
parser.add_argument('-f', '--filename', default=None)
parser.add_argument('--max_jobs', type=int, default=5000, help="limit the maximum number of jobs")
args = parser.parse_args()
callstring = args.callstring.strip().replace('SLURM_ARRAY_TASK_ID', '$SLURM_ARRAY_TASK_ID')

# Parse arguments to get dictionary of experiments

arguments = re.findall('(\\-+[a-zA-Z\_]+)', callstring)
split_points = [callstring.index(a) for a in arguments]
argument_dict = {}
call_head = callstring[:callstring.index(arguments[0])]
for a, i,j in zip(arguments, split_points, split_points[1:] + [len(callstring)]):
    arg = callstring[i+len(a):j].strip()
    if len(arg) > 0:
        if '{' in arg:
            if '-' in arg:
                s, e = [int(i) for i in arg[1:-1].split('-')]
                argument_dict[a] = list(range(s,e+1))
            if ',' in arg:
                argument_dict[a] = [i.strip() for i in arg[1:-1].split(',')]
        else:
            argument_dict[a] = [arg]
    else:
        argument_dict[a] = ['']

keys = argument_dict.keys()

# generate experiment dictionary

experiments = [{k:i for k,i in zip(keys, element)} for element in product(*[v for v in argument_dict.values()])]
call_strings = [" ".join([f'{k} {v}' for k,v in i.items()]) for i in experiments]

if args.local:
    for i, c in enumerate(call_strings):
        print(call_head + c.replace('  ', ' ') + '', file=None)
else:
    if len(call_strings) <= args.max_jobs:
        jobfile = None if args.filename is None else open(args.filename, 'w')
        print(header.format(pwd=os.getcwd(), n_jobs=len(call_strings)), file=jobfile)
        for i, c in enumerate(call_strings):
            print(f'{i+1}) '+call_head + c.replace('  ', ' ') + ';;', file=jobfile)
        print(FOOTER, file=jobfile)
    else:
        for i, cs in enumerate(chunks(call_strings, args.max_jobs)):
            jobfile = None if args.filename is None else open(args.filename + f"{i}.sh", 'w')
            print(header.format(pwd=os.getcwd(), n_jobs=len(cs)), file=jobfile)
            for i, c in enumerate(cs):
                print(f'{i+1}) '+call_head + c.replace('  ', ' ') + ';;', file=jobfile)
            print(FOOTER, file=jobfile)



