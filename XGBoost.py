import Bio
import Bio.motifs as motifs
from Bio import SeqIO
import numpy as np
import xgboost as xgb
import argparse, sys,time,os
import multiprocessing

tic = time.time()

def calculate(seqTup, mnum):
    sequence = chromos[seqTup[0]][seqTup[1]:seqTup[2]]
    sequence = bytes(sequence)
    motif = motifs_list[mnum][0]
    logodds = motifs_list[mnum][1]
    n = len(sequence)
    m = motif.length

    # Create the numpy arrays here; the C module then does not rely on numpy
    # Use a float32 for the scores array to save space
    scores = np.empty(n - m + 1, np.float32)
    ##causes huge slowdown
#     logodds = np.array(
#         [[motif.pssm[letter][i] for letter in "ACGT"] for i in range(m)], float
#     )
    motifs._pwm.calculate(sequence.upper(), logodds, scores)
    
#     threshold = float(motif.name[:].split('\t')[-1])
#     pos_ind = scores >= threshold
#     return sum(pos_ind)
    return scores

def search(seqTup, mnum, chunksize=10**6):
    """Find hits with PWM score above given threshold.
    A generator function, returning found hits in the given sequence
    with the pwm score higher than the threshold.
    """
    seq = chromos[seqTup[0]][seqTup[1]:seqTup[2]]
    seq_len = len(seq)
    motif = motifs_list[mnum][0]
    motif_l = motif.length
    chunk_starts = np.arange(0, seq_len, chunksize)
    for chunk_start in chunk_starts:
        subseq = seq[chunk_start : chunk_start + chunksize + motif_l - 1]
        pos_scores = calculate(seqTup, mnum)
        threshold = float(motif.name[:].split('\t')[-1])
        pos_ind = pos_scores >= threshold
        pos_positions = np.where(pos_ind)[0] + chunk_start
        pos_scores = pos_scores[pos_ind]
        
        neg_positions = np.empty((0), dtype=int)
        neg_scores = np.empty((0), dtype=int)
        chunk_positions = np.append(pos_positions, neg_positions - seq_len)
        chunk_scores = np.append(pos_scores, neg_scores)

        yield from zip(chunk_positions, chunk_scores)


def getFeatures2(seq):
    feats = []
    for m in range(len(motifs_list)):
        #val = calculate(seq, m)
        val = len(list(search(seq, m)))
        feats.append(val)
    return feats

def getlabels(line):
    return line.split()[3]

def process_line2(line):
    vals = line.split()
    c = vals[0]
    start = int(vals[1])
    end = int(vals[2])
    seqInfo = (c,start,end)
    return getFeatures2(seqInfo)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
    parser.add_argument('-g','--genome',help='reference genome FASTA file',required=True)
    parser.add_argument('-t','--train',help='training bed file with chromosome, start, end and label',required=True)
    parser.add_argument('-m','--motifs',help='file of motifs to use as features',required=True)
    parser.add_argument('-p','--predict',help='training bed file for prediction with chromosome, start, end', required=True)
    parser.add_argument('-o','--output_file',help='Output predictions',required=True)

    args = parser.parse_args()
    ###featurization
    chromos = {}
    for seqr in SeqIO.parse(args.genome,'fasta'):
        chromos[seqr.name] = seqr.seq

    motifs_list = []
    with open(args.motifs) as handle:
        for m in motifs.parse(handle, "pfm-four-columns"):
            logodds = np.array([[m.pssm[letter][i] for letter in "ACGT"] for i in range(m.length)], float)
            motifs_list.append((m,logodds))

    train_lines = open(args.train).readlines()
    test_lines  = open(args.predict).readlines()

    pool = multiprocessing.Pool()

    label_str = pool.map(getlabels, train_lines)
    np_label = np.array(label_str)
    train_labels = np_label.astype(float)

    train = pool.map(process_line2, train_lines)
    test  = pool.map(process_line2, test_lines)

    train_feats = np.array(train)
    test_feats = np.array(test)

    ####xgboost
    dtrain = xgb.DMatrix(train_feats, label=train_labels)
    dtest  = xgb.DMatrix(test_feats)

    treeparms = {'max_depth':2,'eta':0.1,'colsample_bytree':0.8,'tree_method':'hist', 'seed':42, 'subsample':0.8}
    xgbt = xgb.train(treeparms,dtrain,200)

    predicts = xgbt.predict(dtest)
    np.save(args.output_file,predicts)

toc = time.time()
print(toc-tic)

