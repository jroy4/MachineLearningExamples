import numpy as np
import argparse
import pyBigWig
from sklearn.utils.extmath import randomized_svd
from glob import glob
from numba import jit

import time


parser = argparse.ArgumentParser(description='Compute NMF to find latent variables with high correlation to transcription start sites')
parser.add_argument('-b','--bw_dir',help='Directory containing bigWig file(s)',required=True)
parser.add_argument('-c','--chromosome_number',help='Which chromosome to get the values for',required=True)
parser.add_argument('-s','--start_pos',type=int,help='Position to start reading from')
parser.add_argument('-e','--end_pos',type=int,help='Position to stop reading at')
parser.add_argument('-k',type=int,default=10,help='Number of latent vectors')
parser.add_argument('-o','--output_file',help='Output file of latent factors matrix.',required=True)

args = parser.parse_args()

tic = time.time()
files = glob(f'{args.bw_dir}/*.bw')

#construct matrix of values from bigWig files
vals = []
for idx, fname in enumerate(files):
    # IMPLEMENT -- use pyBigWig to access the .bw files
	# use args.chromosome_number to access the correct chromosome
	# use args.start_pos and args.end_pos for the start and end position of the chromosome
    bw = pyBigWig.open(fname)
    vals.append(bw.values(args.chromosome_number, args.start_pos, args.end_pos))
    bw.close()
    #deal with NaN, apply any other transformations
matrix = np.array(vals).T
matrix[np.isnan(matrix)] = 0.0

toc = time.time()
print(toc-tic)

#setup proximity operator using the provided code

@jit(nopython=True)
def pyprox_dp(y, lam): #return theta
    n = len(y)
    if n == 0:
        return 

    theta = np.zeros_like(y)    
    # Take care of a few trivial cases
    if n == 1 or lam == 0:
        for i in range(n):
            theta[i] = y[i]
        return theta

            
  # These are used to store the derivative of the
  # piecewise quadratic function of interest
    afirst = 0.0
    alast = 0.0
    bfirst = 0.0
    blast = 0.0
    
    x = np.zeros(2*n)
    a = np.zeros(2*n)
    b = np.zeros(2*n)

  
    l = 0
    r = 0

  # These are the knots of the back-pointers
    tm = np.zeros(n-1)
    tp = np.zeros(n-1)

  # We step through the first iteration manually
    tm[0] = -lam+y[0];
    tp[0] = lam+y[0];
    l = n-1;
    r = n;
    x[l] = tm[0];
    x[r] = tp[0];
    a[l] = 1;
    b[l] = -y[0]+lam;
    a[r] = -1;
    b[r] = y[0]+lam;
    afirst = 1;
    bfirst = -lam-y[1];
    alast = -1;
    blast = -lam+y[1];


  # Now iterations 2 through n-1
    lo = 0
    hi = 0
    alo = 0.0
    blo = 0.0
    ahi = 0.0
    bhi = 0.0
    
    for k in range(1,n-1):
        # Compute lo: step up from l until the
        # derivative is greater than -lam
        alo = afirst
        blo = bfirst
        for lo in range(l,r+1):            
            if alo*x[lo]+blo > -lam: break

            alo += a[lo];
            blo += b[lo];
        else:
            lo = r+1
        
        # Compute the negative knot

        tm[k] = (-lam-blo)/alo
        l = lo-1
        x[l] = tm[k]

        # Compute hi: step down from r until the
        # derivative is less than lam
        ahi = alast;
        bhi = blast;
        for hi in range(r,l-1,-1):
            if -ahi*x[hi]-bhi < lam: break
            ahi += a[hi]
            bhi += b[hi]
        else:
            hi = l-1        

        # Compute the positive knot
        tp[k] = (lam+bhi)/(-ahi);
        r = hi+1;
        x[r] = tp[k];

        # Update a and b
        a[l] = alo;
        b[l] = blo+lam;
        a[r] = ahi;
        b[r] = bhi+lam;



        afirst = 1;
        bfirst = -lam-y[k+1];
        alast = -1;
        blast = -lam+y[k+1];
        

  # Compute the last coefficient: this is where 
  # the function has zero derivative

    alo = afirst;
    blo = bfirst;
    for lo in range(l, r+1):
        if alo*x[lo]+blo > 0: break
        alo += a[lo];
        blo += b[lo];
  
    theta[n-1] = -blo/alo;

  # Compute the rest of the coefficients, by the
  # back-pointers
    for k in range(n-2,-1,-1):
        if theta[k+1]>tp[k]:
            theta[k] = tp[k]
        elif theta[k+1]<tm[k]:
            theta[k] = tm[k]
        else:
            theta[k] = theta[k+1];
  

    return theta

def init_H(Y,k):
    # initialize H
    # can be a random initialization or using the randomized_svd from sklearn
    H = np.random.rand(k, np.size(Y, 1))

    return H

def NMF_FL(Y, k, num_iter=50, l2penalty=1, fl_lambda=1):
    H = init_H(Y,k)
    W = None

    #this is the diagonal offset
    #if l2penalty is small all this does is make the matrix invertible
    D=np.eye(k) * l2penalty
    for n in range(num_iter):
        # Update W
        # $W \leftarrow Y H^T (H H^T + D)^{-1}$
        W = Y @ H.T @ np.linalg.inv(H @ H.T + D)

        # Set negative elements of W to 0
        np.clip(W, a_min=0, a_max=None, out=W)

        # apply fused lasso
        for i in range(W.shape[1]):
            W[:,i] = pyprox_dp(W[:,i], fl_lambda)

        # Update H
        H = np.linalg.inv(W.T @ W + D) @ W.T @ Y

        # Set negative elements of H to 0
        np.clip(H, a_min=0, a_max=None, out=H)

        #early stopping?
    return W, H

# Change BIGWIG_DATA to the name of your value matrix
BIGWIG_DATA = matrix
# num_iter, l2penalty, and fl_lambda are all hyperparameters that should be tuned to maximize correlation with genes
W, H = NMF_FL(BIGWIG_DATA, args.k, num_iter=25, l2penalty=100, fl_lambda=700)

np.save(args.output_file,W,allow_pickle=True)

toc = time.time()
print(toc-tic)