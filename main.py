from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers


CUDA = True
VOCAB_SIZE = 15000
MAX_SEQ_LEN = 40
START_LETTER = 0
BATCH_SIZE = 8
MLE_TRAIN_EPOCHS = 10
ADV_TRAIN_EPOCHS = 10
POS_NEG_SAMPLES = 100000

GEN_EMBEDDING_DIM = 300
GEN_HIDDEN_DIM = 256
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

oracle_samples_path = './oracle_samples.trc'
oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

from tqdm import tqdm
def getLength(line):
    vec=[int(token) for token in line.split()]
    return vec.index(0)
def calculatePPL(gen,testpath):
    testset=loadData(testpath)
    testset_tensor=torch.tensor(testset)

    length=[]
    with open(testpath,'r') as fin:
        for line in fin:
            length.append(getLength(line))
    length=np.array(length)
    
    nll_all=[]
    TEST_SIZE=testset_tensor.shape[0]
    for i in tqdm(range(0, TEST_SIZE)):
        inp, target = helpers.prepare_generator_batch(testset_tensor[i:i + 1], start_letter=START_LETTER,
                                                              gpu=CUDA)
        nll = gen.batchNLLLoss(inp, target)
        nll_all.append(float(nll.data.cpu()))
    nll_all=np.array(nll_all)

    return np.mean(2**(nll_all/length))

def train_generator_MLE(gen, gen_opt, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i in range(0, POS_NEG_SAMPLES, BATCH_SIZE):
            inp, target = helpers.prepare_generator_batch(real_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()
        # Generate LSTM samples
        path='output/MSE-{}.samples'.format(epoch)
        generateSamples(gen, path)

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN


        print(' average_train_NLL = %.4f' % (total_loss))


def train_generator_PG(gen, gen_opt, validation_data_samples, dis, num_batches,_id=0):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # Generate LSTM samples
    path='output/ADV-{}.samples'.format(_id)
    generateSamples(gen, path)
    
    #validation_loss=0
    #VAL_SIZE=validation_data_samples.shape[0]
    #for i in range(0, VAL_SIZE, BATCH_SIZE):
    #    inp, target = helpers.prepare_generator_batch(validation_data_samples[i:i + BATCH_SIZE], start_letter=START_LETTER,
    #                                                  gpu=CUDA)
    #    gen_opt.zero_grad()
    #    loss = gen.batchNLLLoss(inp, target)
    #    validation_loss+= loss
    #helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                   #                                start_letter=START_LETTER, gpu=CUDA)

    # print(' validation_loss = %.4f' % validation_loss)


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, trainset, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    len_sampled=100
    perm=np.random.permutation(trainset.shape[0])
    pos_val=torch.tensor(trainset[perm[:len_sampled]])
    neg_val = generator.sample(len_sampled)
    if CUDA:
        pos_val=pos_val.cuda()
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))
            
from torch import Tensor 

def generateSamples(gen, path):
    with open(path,'w') as fout:
        for i in range(10):
            samples=gen.sample(1000)
            samples=np.array(Tensor.cpu(samples.data))

            for i in range(samples.shape[0]):
                row=[]
                for j in range(samples.shape[1]):
                    row.append(str(samples[i][j]))
                fout.write(' '.join(row)+'\n')

def loadData(filepath):
    ret=[]
    with open(filepath,'r') as fin:
        for line in fin:
            ret.append([int(token) for token in line.split()])
    return np.array(ret)


if __name__=="__main__":
    
    trainset=loadData('./dataset/train.vec')
    validationset=loadData('./dataset/valid.vec')
    testset=loadData('./dataset/test.vec')

    trainset_tensor=torch.tensor(trainset)
    validationset_tensor=torch.tensor(validationset)
    oracle=None

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        #oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        oracle_samples = trainset_tensor.cuda()
        trainset_tensor=trainset_tensor.cuda()
        validationset_tensor=validationset_tensor.cuda()
        ### debug 
    #MLE_TRAIN_EPOCHS=1
    #########

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)
    train_generator_MLE(gen, gen_optimizer, trainset_tensor, MLE_TRAIN_EPOCHS)

    torch.save(gen.state_dict(), pretrained_gen_path)
    gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    d_steps=10

    ######debug######
    # d_steps=1   ###
    #################
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, trainset_tensor, gen, trainset, d_steps, 3)

    torch.save(dis.state_dict(), pretrained_dis_path)
    dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    d_steps=5

    ### debug ##
    # d_step=1####
    ############

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, validationset_tensor, dis, 1, _id=epoch)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, trainset_tensor, gen, trainset, d_steps, 3)

        # Generate seqGAN samples
        path='output/seqGAN-epoch{}.samples'.format(epoch)
        generateSamples(gen, path)