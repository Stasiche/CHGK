import sys

path_to_proj = '/'.join(sys.path[0].split('/')[:-2])
sys.path.append(path_to_proj)

import numpy as np
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed

from src.preprocessing.get_data_generator import get_data_generator
from src.generation.generate import generate
from definitions import SBER_MODEL_SMALL
from typing import TextIO


def process_context(context: str) -> bool:
    generated = generate(model_path, tokenizer_path, f'<s> {context}', max_len, beam_size)
    generated = filter_generation(generated)
    return generated 

def filter_generation(line: str) -> str:
    return line.split(r'</s')[0][4:].strip()

def write_genertaion_to_file(f: TextIO, generation: str, context: str):
    f.write(f'{context}|{generation[len(context):]}\n')


def parse_user_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--context-len', type=int, default=5,
                        help='number of words in contexts')

    parser.add_argument('-g', '--generations-num', type=int, default=20,
                        help='number of generations')
    
    parser.add_argument('-m', '--max-len', type=int, default=150,
                        help='max length of generation (including context)')
    
    parser.add_argument('-b', '--beam-size', type=int, default=5,
                        help='number of beams')
    
    parser.add_argument('-o', '--generations-path', default='generations_default.txt',
                        help='path to the file to save the generations')
    
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                        help="number of parallel jobs")

    return parser.parse_args()

args = parse_user_args()
context_len = args.context_len
generations_num = args.generations_num
max_len = args.max_len
beam_size = args.beam_size
n_jobs = args.jobs
generations_file_path = args.generations_path
model_path = '../../src/training/models/'
tokenizer_path = SBER_MODEL_SMALL

np.random.seed(0)

contexts = []
for row in get_data_generator('../../data/dataset.csv'):
    contexts.append(' '.join(row['question'].split()[:context_len]))

indxs = np.random.choice(len(contexts), generations_num, False)
contexts = [contexts[i] for i in indxs]

jobs = []
with open(generations_file_path, 'w') as f:
    f.write('')

for context in contexts:
    jobs.append(delayed(process_context)(context))
generations = Parallel(n_jobs=n_jobs)(jobs)

with open(generations_file_path, 'a') as f:
    for gen, context in zip(generations, contexts):
        write_genertaion_to_file(f, gen, context)
