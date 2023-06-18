from ipaddress import collapse_addresses
import numpy as np
import pandas as pd
import argparse, time
import sklearn
from sklearn import linear_model, ensemble, svm
from sklearn import metrics
from sklearn import feature_selection, preprocessing

parser = argparse.ArgumentParser(description='Predict if genes are mutated based on mRNA expression data.')
parser.add_argument('-e','--train_expr',nargs='+',help='Expression data file(s)',required=True)
parser.add_argument('-m','--train_mut',nargs='+',help='Mutation data file(s)',required=True)
parser.add_argument('-p','--test_expr',help='Expression data of patients to predict',required=True)
parser.add_argument('-g','--gene',help='Hugo symbol of gene(s) to predict if mutated',nargs='+',required=True)
parser.add_argument('-o','--output_prefix',help='Output prefix of predictions. Each gene\'s predictions are output to PREFIX_GENE.txt',required=True)

args = parser.parse_args()

tic = time.perf_counter()

# reads in the expression file
def read_expr(fname):
    print(fname)
    expr = pd.read_table(fname)

    # perform some kind of normalization here
    scaler = preprocessing.StandardScaler().fit(expr.iloc[:, 2:])
    expr.iloc[:, 2:] = scaler.transform(expr.iloc[:, 2:])
    return expr

def read_mut(fname):
    test = open(fname, "r")
    if test.read(1) == '#':
        return(pd.read_table(fname, skiprows=1,low_memory=False))
    else:
        return(pd.read_table(fname, low_memory=False))
    #return mut

# returns the mutation label
def getLabels(expression, mutation, gene):
    gene_mut = mutation[mutation['Hugo_Symbol'] == gene]

    gene_mut = gene_mut[list(~gene_mut['Consequence'].str.contains('synonymous_variant'))]

    toReturn = pd.DataFrame(np.zeros((expression.shape[1],1)))
    toReturn.index = expression.columns
    toReturn.columns = ['label']
    #toReturn['label'][gene_mut['Tumor_Sample_Barcode'].tolist()] = 1
    toReturn.loc[toReturn.index.isin(gene_mut['Tumor_Sample_Barcode'].tolist())] = 1
    #print(np.where(np.array(toReturn.index.isin(gene_mut['Tumor_Sample_Barcode']))))
    return np.array(toReturn['label'])

# averages the expression of genes that are duplicated in the expression data
def handleDuplicates(expr, gene):
    data = expr[expr.index == gene]
    avg_data = data.mean(axis=0)
    avg_data.name = gene

    expr = expr.drop(gene, axis=0)
    expr = expr.append(avg_data)
    return expr

# Label dependent feature selection
def featureSelection(expr, y, k_num):
    model = feature_selection.SelectKBest(feature_selection.f_regression, k=k_num)
    X_new = model.fit_transform(expr.T, y.T)
    return X_new, model.get_support()

# read in expression training data (X)
expr_all = pd.DataFrame()
index = 0
for fname in args.train_expr:
    if index == 0:
        expr_all = read_expr(fname)
    else:
        expr = read_expr(fname)
        expr_all = expr_all.merge(expr)
        index += 1

expr_all = expr_all.dropna(subset=['Hugo_Symbol'])    
expr_all.index = expr_all['Hugo_Symbol']
expr_all = expr_all.drop(['Hugo_Symbol','Entrez_Gene_Id'], axis=1)

# fill in the NaN values with 0
#expr_all = expr_all.fillna(value = 0)
expr_all = expr_all.fillna(expr_all.mean())
#print(expr_all)

index = 0
mut_all = pd.DataFrame()
# read in mutation training data (Y)
for fname in args.train_mut:
    # read in genes with mutations
    if index == 0:
        mut_all = read_mut(fname)
    else:
        mut = read_mut(fname)
        mut_all = mut_all.merge(mut)

#print(mut_all)    
mut_all = mut_all[['Hugo_Symbol','Tumor_Sample_Barcode','Consequence']]

# Read in the test file
test_file = read_expr(args.test_expr)

test_genes = test_file['Hugo_Symbol'].unique()
test_file.index = test_file['Hugo_Symbol']
test_file = test_file.drop(['Hugo_Symbol','Entrez_Gene_Id'], axis=1)
test_file = test_file.fillna(value = 0)

train_genes = expr_all.index.unique().tolist()
genes_to_keep = list(set(test_genes) & set(train_genes))

expr_all = expr_all.loc[genes_to_keep,:]
test_file = test_file.loc[genes_to_keep,:]

# Handle duplicate genes in the expression and test files
u, c = np.unique(expr_all.index, return_counts = True)
duplicated_expr_genes = u[c > 1]
for gene in duplicated_expr_genes:
    expr_all = handleDuplicates(expr_all, gene)

u, c = np.unique(test_file.index, return_counts = True)
duplicated_test_genes = u[c > 1]
for gene in duplicated_test_genes:
    test_file = handleDuplicates(test_file, gene)

for gene in args.gene:
    # label-dependent feature selection
    mut_y = getLabels(expr_all, mut_all, gene)
    #X_train, feature_mask = featureSelection(expr_all, mut_y, k_num = expr_all.shape[1])
    X_train, feature_mask = featureSelection(expr_all, mut_y, k_num = 800)

    #train model
    model = linear_model.LogisticRegression(C=0.0001, max_iter=10000).fit(X_train, mut_y)

    genes_to_keep = expr_all.index[feature_mask]
    X_test = test_file.loc[genes_to_keep]

    #predict
    output = model.predict_proba(X_test.values.T)

    final_output = zip(list(X_test.columns), output[:,1])
    out = open('%s_%s.txt'%(args.output_prefix,gene),'wt')
    for (name,p) in sorted(final_output):
        out.write('%s %.5f\n'%(name,p))
        
toc = time.perf_counter()
print('Time Elapsed: {}'.format(toc-tic))