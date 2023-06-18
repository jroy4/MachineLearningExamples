import os
import random
import time
from sklearn import preprocessing,metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import xgboost as xgb
from scipy import stats

import cloudpickle 

def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient
    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])
    Output:  rs     Spearman's rank correlation coefficient
    """

    rs = stats.spearmanr(y, f)[0]

    return rs

def get_stand_inchi(smile):
    key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smile))

    return key

# This is the training data David gave us
chembl_activities=pd.read_csv('chembl_activities.txt', sep=' ')
chembl_activities=chembl_activities.rename(columns={"target": "Chembl",})
chembl_activities

# This is the mapping between chmbl and uniprot IDs
chembl_uniprot_mapping = pd.read_csv('chembl_uniprot_mapping.txt', sep='\t', header=None, comment='#')
chembl_uniprot_mapping.columns = ['UniProt','Chembl', 'Name', "Type"]
chembl_uniprot_mapping = chembl_uniprot_mapping[chembl_uniprot_mapping['Type'] == 'SINGLE PROTEIN']
chembl_uniprot_mapping

# These are all proteins that are kinases
kinases = pd.read_csv('kinase_seqs.txt', sep=' ', header=None)
kinases.columns = ['UniProt','Seq']
kinases

# These are proteins with their k seperated bigrams pssm
ksbp = pd.read_csv('k-sep_normalized.tsv', sep='\t')
ksbp = ksbp.rename(columns={"target_id": "UniProt",})
ksbp

# merge the chmbl IDs and the uniprot into a single table
all_proteins = pd.merge(chembl_activities, chembl_uniprot_mapping, how='inner', on=['Chembl'])

#filter the training data so we only have proteins that are kinases
train_kinases = pd.merge(all_proteins, kinases, how='inner', on=['UniProt'])
train_kinases_abr = train_kinases[['smiles','pchembl', 'UniProt', 'Seq']]

#only keep the values I have protein feature data for
train_kinases_abr = train_kinases_abr[train_kinases_abr['UniProt'].isin(ksbp['UniProt'])]
train_kinases_abr

PROT_FEAT_SIZE = 400
def prot_to_ksbpssm(protein):
    exp = ksbp.loc[ksbp['UniProt'] == protein]
    vals = exp.loc[:, exp.columns != 'UniProt'].values
    tempReturn= np.zeros(PROT_FEAT_SIZE)
    if vals.shape[0] != 1:
        print(protein)
        return tempReturn
    for i in range(tempReturn.shape[0]):
        tempReturn[i] = vals[0][i]
    return tempReturn


# all_proteins_ksbp = np.zeros((train_kinases_abr['UniProt'].shape[0], PROT_FEAT_SIZE), dtype=np.int8)
# for ind, protein in enumerate(train_kinases_abr['UniProt']):
#     all_proteins_ksbp[ind] = prot_to_ksbpssm(protein)
#     if ind % round(TRAIN_SIZE/20) == 0:
#         print('Processing...{}% complete'.format(ind*100/307843))
# print('Done!')

MORGAN_SIZE = 1024
radius = 2
def smi_to_morganfingerprint(smi, radius, MORGAN_SIZE):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        tempReturn = np.zeros(MORGAN_SIZE, dtype=np.int8)
        vec = AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=MORGAN_SIZE)
        for i in range(tempReturn.shape[0]):
            tempReturn[i] = vec[i]   
        return tempReturn
    else:
        return np.zeros(MORGAN_SIZE)

    
# all_SMILES_MORGAN = np.zeros((train_kinases_abr['smiles'].shape[0], MORGAN_SIZE), dtype=np.int8)
# for ind, smi in enumerate(train_kinases_abr['smiles']):
#     all_SMILES_MORGAN[ind] = smi_to_morganfingerprint(smi, radius, MORGAN_SIZE)
#     if ind % round(TRAIN_SIZE/20) == 0:
#         print('Processing...{}% complete'.format(ind*100/307843))
# print('Done!')

def featurize(data):
    DATA_SIZE = data.shape[0]
    DATA_SIZE_FIVE_PERCENT = round(DATA_SIZE / 25)
    
    
    all_proteins_ksbp = np.zeros((data['UniProt'].shape[0], PROT_FEAT_SIZE), dtype=np.float32)
    print('Featurizing proteins...')
    for ind, protein in enumerate(data['UniProt']):
        all_proteins_ksbp[ind] = prot_to_ksbpssm(protein)
        if ind % DATA_SIZE_FIVE_PERCENT == 0:
            print('\tProcessing...{}% complete'.format(round(ind*100/DATA_SIZE)))
    print('\tDone!\n')
    
    print('Featurizing SMILES...')
    all_SMILES_MORGAN = np.zeros((data['smiles'].shape[0], MORGAN_SIZE), dtype=np.float32)
    for ind, smi in enumerate(data['smiles']):
        all_SMILES_MORGAN[ind] = smi_to_morganfingerprint(smi, radius, MORGAN_SIZE)
        if ind % DATA_SIZE_FIVE_PERCENT == 0:
            print('\tProcessing...{}% complete'.format(round(ind*100/DATA_SIZE)))
    print('\tDone!')
    
    together = np.concatenate([all_proteins_ksbp,all_SMILES_MORGAN],axis=1)
    return together


training1 = featurize(train_kinases_abr)
training1

# remove the training data with NaN values
# training = np.concatenate([all_proteins_ksbp,all_SMILES_MORGAN],axis=1)
training = training1
training = training[~np.isnan(training).any(axis=1)]
y_training = np.array(train_kinases_abr[~np.isnan(training).any(axis=1)]['pchembl'])

testFile  = 'indep.txt'
validFile1 = 'test1_labels.txt'
validFile2 = 'test2_labels.txt'


valid_table = pd.read_csv(validFile1, sep=' ')
valid_table = valid_table.rename(columns={'SMILES':'smiles'})
valid_table
valid1 = featurize(valid_table)
valid1_labels = np.array(valid_table['pKd'])

valid2_table = pd.read_csv(validFile2, sep=' ')
valid2_table = valid2_table.rename(columns={'SMILES':'smiles'})
valid2_table
valid2 = featurize(valid2_table)
valid2_labels = np.array(valid2_table['pKd'])


clf=RandomForestRegressor(random_state=42, n_estimators=4, max_samples = 0.33)
clf.fit(training,training_labels)

y_predict = clf.predict(valid1)
print(spearman(valid1_labels, y_predict))

y_predict = clf.predict(valid2)
print(spearman(valid2_labels, y_predict))

cloudpickle.dump(clf,open('model.pkl','wb'))

