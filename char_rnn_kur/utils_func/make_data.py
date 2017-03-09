"""
Copyright 2016 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from vocab import *
import json
import os

                                                                    # create data/
if not os.path.exists('./data/'):
    os.mkdir('./data/')

    
                                                                    # create one_hot function
                                                                    # v: list of char index for a corpose
                                                                    # ndim: n_vocab
def one_hot(v, ndim):
    v_one_hot = np.zeros(
        (len(v), ndim,)
    )
    for i in range(len(v)):
        v_one_hot[i][v[i]] = 1.0
    return v_one_hot

x = []
y = []

                                                                    # join 2 or more book txt files into a big list of 

all_chars = []
for book in [
    'pride_and_prejudice.txt',                                       
    'shakespeare.txt'
]:
    with open('books/%s' % book, 'r') as infile:
        chars = [
            c for c in ' '.join(infile.read().lower().split())
            if c in set(vocab)
        ]
                                                                    # make sure all chars are restricted to 30 unqiue chars
        
        all_chars += [' ']
        all_chars += chars

all_chars = list(' '.join(''.join(all_chars).split()))
num_chars = len(all_chars)
with open('cleaned.txt', 'w') as outfile:
    outfile.write(''.join(all_chars))                 
                                                                     # join all words into a long string


x, y = [], []

data_portions = [                                                     # split datasets 
    ('train', 0.8),
    ('validate', 0.05),
    ('test', 0.05),
    ('evaluate', 0.05),
]

dev = True                                                             # False: use all data
if dev:
    # shrink data to make things go faster
    for i in range(len(data_portions)):
        data_portions[i] = (
            data_portions[i][0],
            data_portions[i][1] * 0.1
        )
        
        
                                                                        # max_i = sum(num_train_char, num_validate_char, ...)
                                                                        #            - seq_len
max_i = sum([
    int(round(len(all_chars) * fraction))
    for name, fraction in data_portions
]) - seq_len




                                                                # take all_chars
                                                                # use seq_len as a window, shift 1 char to right at a time
                                                                # each window is a sample (1, 50)
                                                                # one-hot transform to each sample 
                                                                # now, each sample for x, dim (50, 30)
                                                                # now, each sample for y, dim (1, 30) or (30, )

for i in range(max_i):

    in_char_seq = all_chars[i: i + seq_len]

    # one hot representation
    sample_x = np.zeros((len(in_char_seq), n_vocab,))
    for j, c in enumerate(in_char_seq):
        sample_x[j][char_to_int[c]] = 1
    x.append(sample_x)
                                                                # x is a list, append each sample_x into x, as an element
    sample_y = np.zeros(n_vocab)
    sample_y[char_to_int[all_chars[i + seq_len]]] = 1
    y.append(sample_y)
                                                                # y is a list, append each sample_y into y, as an element

    
                                                                # transform x, y from list to np.array
x, y = np.array(x).astype('int32'), np.array(y).astype('int32')


                                                                # each sample (50, 30)
                                                                # len(x) is num of samples in x
                                                                # split x by proportion for train, validate, evalute, test parts
                                                                # split y by proportion for train, validate, evalute, test parts
                                                                # save them into jsonl
start_i = 0
for name, fraction in data_portions:
    end_i = start_i + int(round(len(x) * fraction))
                
                                                                           # print a name
                                                                           # print sample index: 
    print(name, ":")  # 
    print("sample Index:", start_i, end_i)     # 
    x0 = x[start_i: end_i]
    y0 = y[start_i: end_i]

    print('dims:')
    print(x0.shape)
    print(y0.shape)

    start_i = end_i

    with open('data/%s.jsonl' % name, 'w') as outfile:
        for sample_x, sample_y in zip(x0, y0):
            outfile.write(json.dumps({
                'in_seq': sample_x.tolist(),
                'out_char': sample_y.tolist()
            }))
            outfile.write('\n')

    del x0, y0
