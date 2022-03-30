'''
Update:  
Author: Huong N. Pham
Classification problem: BreastMass

'''
from pyLib.sendEmail import send_email
from pyLib.stopInstance import stop_instance
import os
import time
import pickle
import shutil
import argparse
import numpy as np
from random import randrange

import copy
import torch
import torchvision


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets.folder import ImageFolder

#################################################################
# Default parameters
'''

'''
#################################################################
class ImageFolderWithIDs(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithIDs, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path

#"/content/drive/My Drive/Colab Notebooks/Kaggle/DME/val/"
def load_images(directory):
    from torchvision import transforms
    transforms = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.5394, 0.5394, 0.5394],std=[0.2447, 0.2447, 0.2447])])
    data = ImageFolderWithIDs(root=directory, transform=transforms)

#    test_data_path = "/content/drive/My Drive/Colab Notebooks/Kaggle/DogCat/sample/test/"
#    test_data = ImageFolderWithIDs(root=test_data_path, transform=transforms)
    return data
def get_train_valid_test_loader(args, random_seed, augment = False, valid_size=0.2, test_size=0.1, shuffle=True):
    
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.5394, 0.5394, 0.5394],
        std=[0.2447, 0.2447, 0.2447],
    )

    # define transforms
    valid_transform = transforms.Compose([
                      transforms.Resize([224,224]),
                      transforms.ToTensor(),
                      normalize,
                                        ])
    test_transform  = transforms.Compose([
                      transforms.Resize([224,224]),
                      transforms.ToTensor(),
                      normalize,
                                        ])
    if augment:
        train_transform = transforms.Compose([
                      transforms.Resize([224,224]),
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize,
        ])
    else:
        train_transform = transforms.Compose([
                          transforms.Resize([224,224]),
                          transforms.ToTensor(),
                          normalize,
        ])

    # load the dataset with the whole data
    train_dataset_transform = ImageFolderWithIDs(root=args.data_path, transform=train_transform)
    valid_dataset_transform = ImageFolderWithIDs(root=args.data_path, transform=valid_transform)
    test_dataset_transform  = ImageFolderWithIDs(root=args.data_path, transform=test_transform)

    num_train = len(train_dataset_transform)
    indices = list(range(num_train))
    split_valid = int(np.floor(valid_size * num_train))
    split_test = int(np.floor(test_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx, test_idx = indices[(split_valid+split_test):], indices[split_test:(split_valid+split_test)], indices[:split_test]

    train_dataset = torch.utils.data.Subset(train_dataset_transform, train_idx)
    valid_dataset = torch.utils.data.Subset(valid_dataset_transform, valid_idx)
    test_dataset  = torch.utils.data.Subset(test_dataset_transform , test_idx)
    
    return (train_dataset, valid_dataset, test_dataset)

'''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
        # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=shuffle,)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)
'''


    

def extract_data(args):
    '''
    Translate Image data structure into a data set for training/evaluating a single model
    
    @param args Argparse object, which contains key information, including train_datapath, 
            val_data_path, test_data_path
            
    @return Tensor for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of data paths that have been chosen
    '''
    from torchvision import transforms
    # Load data from tensors or images and transforms data to tensor with IDs atttached

    train_dataset, valid_dataset, test_dataset = get_train_valid_test_loader(args,
                                                                            50,
                                                                            False,
                                                                            0.2,
                                                                            0.1,
                                                                            True
                                                                            )
    """test_data_path = "/content/drive/My Drive/Colab Notebooks/Kaggle/DogCat/sample/test/"
    test_dataset = ImageFolderWithIDs(root=test_data_path, transform=transforms)"""

    return train_dataset, valid_dataset, test_dataset

def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
      # Case where no args are given (usually, because we are calling from within Jupyter)
      #  In this situation, we just use the default arguments
      parser = create_parser()
      args = parser.parse_args([])
    
    # Extract the data sets
    train_dataset, val_dataset, test_dataset = extract_data(args)
    
    # Load model
    transfer_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

    # Freeze parameters
    for name, param in transfer_model.named_parameters():
      if("bn" not in name):
        param.requires_grad = False

    # Replace last layer
    transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,3))
    
    # Tune the model
    train(args, transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_dataset, val_dataset, test_dataset)

    # Report if verbosity is turned on
    """if args.verbose >= 1:
        print(model.summary())"""
def train(args, model, optimizer, loss_fn, train_dataset, val_dataset, test_dataset):
    # create file name
    fbase = generate_fname(args)
    ranID = randrange(100000)
    
    #check if gpu is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weight to GPU
    if torch.cuda.is_available():
        model.cuda()
    #weight sampler
    weights = np.ones(len(train_dataset))

    #train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    listRemove = []
    
    training_loss_log = []
    validation_loss_log = []
    accuracy_log = []

    best_acc1 = 0
    accuracy = 0
    
    results = {}
    
    for epoch in range(int(args.epochs)):
        batchNumber = 0
        for i in range(0,args.batchNumber):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()

            sampler = WeightedRandomSampler(weights, args.batch_size)
            train_loader = DataLoader(train_dataset, shuffle=(sampler is None),sampler=sampler, batch_size = args.batch_size)
            
            batch = iter(train_loader).next()
            batchNumber += 1

            optimizer.zero_grad()
            inputs, targets, ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs) 
            print("######################## Epoch {} - Batch {} ########################".format(epoch, batchNumber))
            print("IDs in batch {}: {}".format(batchNumber,ids))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= batchNumber
        # Validation
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets, ids = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(outputs,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(outputs, dim = 1), dim=1)[1],
            targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

            #writer.add_scalar('accuracy', num_correct / num_examples, epoch)
            
        valid_loss /= len(val_loader)
        accuracy = num_correct / num_examples

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, accuracy))

        if accuracy > best_acc1 and accuracy >= 0.68:
            best_acc1 = accuracy
            best_model_state1 = copy.deepcopy(model.state_dict())
            print("Save best Model_1 @ epoch {} acc: {}".format(epoch, best_acc1))
            
            epochState1 = epoch
            best_valid_loss = valid_loss
            best_optimizer = optimizer.state_dict()
            # Save model when achieve the best
            torch.save({
                'best_model': best_model_state1,
                'optimizer'        : best_optimizer,
                'valid_loss_min'   : best_valid_loss,
                'accuracy'         : best_acc1,
                'epoch'            : epochState1
                }, "%smodel"%(fbase))
            send_email("Save 3_class: Max valid accuracy: {} @ epoch {} GPU".format(best_acc1,epochState1))
        if epoch == 2998:
            model_2998 = copy.deepcopy(model.state_dict())
            torch.save({'model_2998': model_2998}, "model_2998")
        # Performance log data
        training_loss_log.append(round(training_loss,2))
        validation_loss_log.append(round(valid_loss,2))
        accuracy_log.append(round(accuracy,2))

        if accuracy >= args.probsThresthold:
          break

    # Generate log data
    
    """results['args'] = args"""

    results['training_loss_log'] = training_loss_log
    results['validation_loss_log'] = validation_loss_log
    results['accuracy_log'] = accuracy_log

    
    results['best_acc'] = [best_acc1]
    results['epochState'] = [epochState1]
    
    # Save log files
    
    print("Saved file as %s_%s_%s.pkl"%(os.path.basename(__file__)[:-3],fbase,ranID))
    results['fname_base'] = fbase
    fp = open("%s_%s_%s.pkl"%(os.path.basename(__file__)[:-3],fbase,ranID), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Show results
    print("Validation accuracy state 1: {} @ epoch {} ".format(best_acc1, epochState1))
    print(accuracy_log)
   

    send_email("{}\nMax validation accuracy: {} @ epoch {} GPU - DONE".format(fbase,best_acc1,epochState1))
    stop_instance('inlaid-fuze-338203','us-central1-a','pytorch-gpu')

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='DME classification')
 
    parser.add_argument('-probsThresthold', type=float, default=0.8, help="probability threshold min to remove out of training ")
    parser.add_argument('-data_path', type=str, default='/home/hpham/Data/DME/train', help='train directory')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-batchNumber', type=int, default=20, help='number of batches to keep training')
    parser.add_argument('-epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('-version', type=int, default=0, help='version')
    return parser
def generate_fname(args):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    '''
    if args.probsThresthold is None:
        probsThresthold_str = ''
    else:
        probsThresthold_str = 'probsThresthold_%0.2f_'%(args.probsThresthold)

    if args.batch_size is None:
        batch_size_str = ''
    else:
        batch_size_str = 'batch_size_%d_'%(args.batch_size)

    if args.batchNumber is None:
        batchNumber_str = ''
    else:
        batchNumber_str = 'batchNumber_%d_'%(args.batchNumber)

    if args.epochs is None:
        epochs_str = ''
    else:
        epochs_str = 'epochs_%d_'%(args.epochs)
    if args.version is None:
        version_str = ''
    else:
        version_str = 'ver_%d_'%(args.version)

    # Put it all together, including #of training folds and the experiment rotation
    return "%s%s%s%s%s"%(
                      probsThresthold_str, 
                      batch_size_str, batchNumber_str, epochs_str, version_str)
        
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)
# nohup python3 breast.py -probsThresthold 0.95 -batchNumber 1 -batch_size 16 -epochs 4000 -data_path './breast' > 3_224.log &