#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F
import os.path

import sys
from sklearn import preprocessing
import imageio
import numpy as np
import argparse, time
import fsspec

import random
from sklearn import metrics
from torchvision import models

#import wandb
#wandb.init(project="assign6", entity="jor115")

torch.manual_seed(42)

labelmap = {
"Actin_disruptors": 0,
"Aurora_kinase_inhibitors": 1,
"Cholesterol-lowering": 2,
"DMSO": 3,
"DNA_damage": 4,
"DNA_replication": 5,
"Eg5_inhibitors": 6,
"Epithelial": 7,
"Kinase_inhibitors": 8,
"Microtubule_destabilizers": 9,
"Microtubule_stabilizers": 10,
"Protein_degradation": 11,
"Protein_synthesis": 12
}



class ImgDataset(torch.utils.data.Dataset):
    '''Dataset for reading in images from a training directory'''
    def __init__(self, args):
        '''Initialize dataset by reading in image locations'''
        with fsspec.open_files(args.train_data_dir+'/TRAIN',mode='rt')[0] as f:
            self.examples = [] # list of (label, [red,green,blue files])
            n_classes = len(labelmap)
            for line in f:
                label, c1, c2, c3 = line.rstrip().split(' ')
                #create absolute paths for image files
                self.examples.append((labelmap[label], [ args.train_data_dir + '/' + c for c in (c1,c2,c3)]))
      
    def open_image(self,path):
        '''Return img at path, caching downloaded images'''
        fname =  path.rsplit('/',1)[-1]
        if path.startswith('gs://'): # check for downloaded file
            if os.path.exists(fname):
                path = fname
        if path.startswith('gs://'): #cache download
            with fsspec.open_files(path,mode='rb')[0] as img:
                out = open(fname,'wb')
                out.write(img.read())
                out.close()
                path = fname
        return  imageio.imread(open(path,'rb'))

    def __len__(self):
        return len(self.examples)
      
    def __getitem__(self, idx):
        imgs = [self.open_image(fname) for fname in self.examples[idx][1]]
        #perhaps consider applying an image transform
        randrotate = np.random.random_sample()
        randflip   = np.random.random_sample()

        if randrotate < 0.5:
            numRot = random.randrange(1, 4)
            imgs = [np.rot90(img, k=1) for img in imgs]
        if randflip < 0.5:
            imgs = [np.fliplr(img) for img in imgs]
#             if(np.random.random_sample() < 0.5):
#                 imgs = [np.fliplr(img) for img in imgs]
#             else:
#                 imgs = [np.flipud(img) for img in imgs]
        return {
            'img': torch.Tensor(np.array(imgs,np.float32)),
            'label': self.examples[idx][0]
            } 


# #define my network - these are not necessarily reasonable hyperparameters
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
#         self.pool1= nn.MaxPool2d(kernel_size=4)
#         self.conv2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3)
#         self.pool2= nn.MaxPool2d(kernel_size=4)
#         self.d1 = nn.Linear(in_features=61504,out_features=64)

#         self.d2 = nn.Linear(in_features=64,out_features=13)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool2(x)
#         x = torch.flatten(x,start_dim=1)
#         x = self.d1(x)
#         x = F.relu(x)
#         x = self.d2(x)
#         #For evaluation we want a softmax - you must return this as
#         #the first element of a tuple.  For training, in order to use
#         #the numerically more stable cross_entropy loss we will also return
#         #the un-softmaxed values
#         return F.softmax(x,dim=1),x
    
#define my network
class MyModel(models.densenet.DenseNet):
    def __init__(self):
        #densenet169
        #growth_rate,, block_config, num_init_features, pretrained, progress
        super(MyModel, self).__init__(32, (6, 12, 32, 32), 64, True, True)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return F.softmax(out,dim=1), out

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses  
    numImages = images.__len__()
    for i in range(numImages):
        count[images.__getitem__(i)['label']] += 1 
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))
    for i in range(nclasses): 
        if not count[i] == 0:
            weight_per_class[i] = N/float(count[i])                               
    weight = [0] * numImages                                             
    for i in range(numImages):                                         
        weight[i] = weight_per_class[images.__getitem__(i)['label']]                                  
    return weight 

def run_training(args):

    #Read the training data
    dataset = ImgDataset(args)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    trainData, validData = torch.utils.data.random_split(dataset,[train_size, test_size])

    epoch_length = args.max_epochs
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(trainData, len(labelmap)))
    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(trainData))
    dataloader = torch.utils.data.DataLoader(trainData,batch_size=20, sampler=weighted_sampler)
  
    # Create an instance of the model
    model = MyModel().to('cuda')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(labelmap))
    model = model.to('cuda')

    model.train()
    loss_object = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    losses = []
    for e in range(args.max_epochs):
        start = time.time()
        for i,batch in enumerate(dataloader):
            optimizer.zero_grad()
            labels = batch['label'].to('cuda')
            x,output = model(batch['img'].to('cuda'))
            loss= F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Write the summaries and print an overview fairly often.
            if i % 50 == 0: #this is too often
                # Print status to stdout.
                print('Epoch %d Step %d: loss = %f' % (e,i, losses[-1]))
                sys.stdout.flush()
        print("Epoch time:",time.time()-start)
        start = time.time()
    # Export the model so that it can be loaded and used later for predictions.
    # For maximum compatibility export a trace of an application of the model
    # https://stackoverflow.com/questions/59287728/saving-pytorch-model-with-no-access-to-model-class-code  
    testdataloader = torch.utils.data.DataLoader(dataset,batch_size=1) #one example at a time for testing
    testbatch = next(iter(testdataloader))
    with torch.no_grad():
        model.eval()
        traced = torch.jit.trace(model, testbatch['img'].to('cuda'))

    torch.jit.save(traced,args.out)
    
    valid_dataloader = torch.utils.data.DataLoader(validData,batch_size=1)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in valid_dataloader:
            pred = model(batch['img'].to('cuda'))
            # the first element of the returned tuple should be the 
            # softmaxed class probabilities
            y_pred.append(np.argmax(pred[0].cpu().numpy()))
            y_true.append(batch['label'][0].item())
    accuracy = metrics.accuracy_score(y_true,y_pred)
    print('Accuracy: {}'.format(accuracy))



if __name__ == '__main__':
    # Basic model parameters as external flags.
    parser = argparse.ArgumentParser('Train a model.')
    parser.add_argument('--max_epochs', default=1, type=int, help='Maximum number of epochs to train.')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size.')
    parser.add_argument('--train_data_dir', default='gs://mscbio2066-data/trainimgs', help='Directory containing training data')
    parser.add_argument('--out', default='model.pth', help='File to save model to.')

    # Feel free to add additional flags to assist in setting hyper parameters

    args = parser.parse_args()
    run_training(args)
