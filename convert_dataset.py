#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle

with open('lenta_short_sentences/lenta.train.txt') as fin:
    train_txt=fin.read()
with open('lenta_short_sentences/lenta.valid.txt') as fin:
    valid_txt=fin.read()
with open('lenta_short_sentences/lenta.test.txt') as fin:
    test_txt=fin.read()
    
alltext=train_txt+' '+valid_txt+' '+test_txt
tokens=set(alltext.split())

# dictionary
dictionary={token:str(i+2) for i,token in enumerate(tokens)}
dictionary['<pad>']=1
dictionary['<start>']=0
pickle.dump(dictionary,open('dataset/dictionary.pic','wb'))
print('len dictionary',len(dictionary))
# 
maxlen=40
def getVector(sentence,dictionary,maxlen):
    sent2vec=[dictionary[token] for token in sentence.split()]
    return sent2vec+['0']*(maxlen-len(sent2vec))
def checkLen(sent, maxlen):
    if sent.startswith('об этом'): return False
    return len(sent.split())<=maxlen
def convertSet(text,dictionary, maxlen):
    return '\n'.join([' '.join(getVector(sent, dictionary, maxlen)) for sent in text.split('\n') if checkLen(sent, maxlen)])
with open('dataset/train.vec','w') as fout:
    fout.write(convertSet(train_txt,dictionary,maxlen))
with open('dataset/valid.vec','w') as fout:
    fout.write(convertSet(valid_txt,dictionary, maxlen))
with open('dataset/test.vec','w') as fout:
    fout.write(convertSet(test_txt,dictionary,maxlen))
