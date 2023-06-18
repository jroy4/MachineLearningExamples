#!/usr/bin/env python3    
import numpy as np
import umap
import faiss
import scanpy
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ignore warning about files we are using

parser = argparse.ArgumentParser(description='Compute clustering using UMAP and FAISS')
parser.add_argument('-d','--data',help='anndata input file',required=True)
parser.add_argument('-k',type=int,help='Number of clusters to identify',required=True)
parser.add_argument('-o','--output_file',help='Output file of cluster assignments (npy).',required=True)

args = parser.parse_args()

#read and preprocess input file
adata = scanpy.read(args.data)
scanpy.pp.filter_genes(adata, min_counts=10000, min_cells=None, max_counts=None, max_cells=None, inplace=True, copy=False)
adata_df = adata.to_df()

#compute umap embedding (you determine a good dimension, probably not 2 or k)
reducer = umap.UMAP(random_state=42)
standard_embedding = reducer.fit_transform(adata_df)

#kmeans cluster
d=standard_embedding.shape[1]
k = args.k
n_init = 10
max_iter = 300
kmeans = faiss.Kmeans(d=d, k=k, niter=max_iter, nredo=n_init)
kmeans.train(standard_embedding.astype(np.float32))

#extract cluster ids into I, a flat one-dimensional array
#extract cluster ids into I, a flat one-dimensional array
D, I = kmeans.index.search(standard_embedding, 1)
I = I.flatten()

np.save(args.output_file,I,allow_pickle=True)