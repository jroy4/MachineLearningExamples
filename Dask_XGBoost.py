import Bio
import Bio.motifs as motifs
from Bio import SeqIO
import numpy as np
import xgboost as xgb
import argparse, sys,time,os
import multiprocessing
import dask
import dask.dataframe as dd
import dask.array as da
import fsspec
from dask_yarn import YarnCluster
from dask.distributed import Client


def calculate(seqLit, mnum):
    sequence = seqLit
    sequence = bytes(sequence, "ASCII")
    motif = motifs_list[mnum][0]
    logodds = motifs_list[mnum][1]
    n = len(sequence)
    m = motif.length
    scores = np.empty(n - m + 1, np.float32)
    motifs._pwm.calculate(sequence.upper(), logodds, scores)
    return scores

def search(seqLit, mnum, chunksize=10**6):
    """Find hits with PWM score above given threshold.
    A generator function, returning found hits in the given sequence
    with the pwm score higher than the threshold.
    """
    seq = seqLit
    seq_len = len(seq)
    motif = motifs_list[mnum][0]
    # motif_l = motif.length
    pos_scores = calculate(seqLit, mnum)
    threshold = float(motif.name[:].split('\t')[-1])
    pos_ind = pos_scores >= threshold
    pos_positions = np.where(pos_ind)[0]
    pos_scores = pos_scores[pos_ind]

#     neg_positions = np.empty((0), dtype=int)
#     neg_scores = np.empty((0), dtype=int)
#     chunk_positions = np.append(pos_positions, neg_positions - seq_len)
#     chunk_scores = np.append(pos_scores, neg_scores)

    return zip(pos_positions, pos_scores)

#only for training
def getfeatures(seqList):
    numRows = seqList.shape[0]
    numCols = len(motifs_list) + 1
    all_feats = np.zeros((numRows, numCols))
    for i, row in seqList.iterrows():
        seq = row['seq']
        label = row['val']
        feats = []
        for m in range(len(motifs_list)):
            val = len(list(search(seq, m)))
            feats.append(val)
        feats.append(label)
        all_feats[i] = feats
    return np.array(all_feats)

#only for testing
def getfeaturesTest(seqList):
    numRows = seqList.shape[0]
    numCols = len(motifs_list)
    all_feats = np.zeros((numRows, numCols))
    for i, row in seqList.iterrows():
        seq = row['seq']
        feats = []
        for m in range(len(motifs_list)):
            val = len(list(search(seq, m)))
            feats.append(val)
        all_feats[i] = feats
    return np.array(all_feats)


if __name__ == '__main__':
    tic = time.time()

    parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
    parser.add_argument('-t','--train',help='training bed file with chromosome, start, end and label',required=True)
    parser.add_argument('-m','--motifs',help='file of motifs to use as features',required=True)
    parser.add_argument('-p','--predict',help='training bed file for prediction with chromosome, start, end', required=True)
    parser.add_argument('-o','--output_file',help='Output predictions',required=True)

    args = parser.parse_args()

    threads = 1
    mem = '3.57GiB'
    workers = 480
    cluster = YarnCluster(worker_vcores=threads,worker_memory=mem,n_workers=workers)
    client = Client(cluster)

    print('I started :)')
    motif_source = args.motifs
    motifs_list = []
    with fsspec.open(motif_source, 'rt') as handle:
        for m in motifs.parse(handle, 'pfm-four-columns'):
            logodds = np.array([[m.pssm[letter][i] for letter in "ACGT"] for i in range(m.length)], float)
            motifs_list.append((m,logodds))


    df_train = dask.dataframe.read_csv(args.train,   header=None, delim_whitespace=True, blocksize='0.2495MB', names=('seq','val'))
    df_test  = dask.dataframe.read_csv(args.predict, header=None, delim_whitespace=True, blocksize='0.0269MB', names=['seq'])
    print('I\'m doing stuff :)')
    
    features_and_label = df_train.map_partitions(getfeatures, meta=np.zeros((2,863)))
    features_and_label = features_and_label.persist()
    features_and_label.compute_chunk_sizes()
    features_and_label = features_and_label.rechunk((30738, -1))
    features = features_and_label[:,:-1]
    labels = features_and_label[:,-1:]
    print('Time to featurize: {}'.format(time.time()-tic))
    testfeats = df_test.map_partitions(getfeaturesTest, meta=np.zeros((2,862)))
    testfeats = testfeats.persist()
    testfeats.compute_chunk_sizes()
    testfeats = testfeats.rechunk((3436,-1))
    
    tic1 = time.time()
    dtrain = xgb.dask.DaskDMatrix(client, features, labels)
    treeparms = {'max_depth':2,'eta':0.15,'colsample_bytree':0.8,'tree_method':'hist', 'seed':42, 'subsample':0.8, 'nthread':32}
    output = xgb.dask.train(
        client,
        treeparms,
        dtrain,
        num_boost_round=145,
        evals=[(dtrain, "train")],
    )
    print('Time to train: {}'.format(time.time()-tic1))
    y_hat = xgb.dask.predict(client, output, testfeats).persist()
    y_hat.compute_chunk_sizes()
    np.save(args.output_file, y_hat)
    toc = time.time()
    print(toc-tic)