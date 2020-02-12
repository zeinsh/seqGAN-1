#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cPickle

dictionary=cPickle.load(open('dataset/dictionary.pic','rb'))
idx2token={idx:token for token,idx in dictionary.items()}

with open('save/generated_text_seqgan.txt','w') as fout:
    with open('save/generator_sample.txt','r') as fin:
        for line in fin:
            ret=' '.join([idx2token.get(token,'<out>') for token in line.strip().split() if token!='0'])+'\n'
            fout.write(ret)
