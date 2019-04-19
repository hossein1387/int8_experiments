import argparse
import os
import sys
import torchvision.datasets as dsets
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torchvision.transforms as transforms
import os.path
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False, default="config.yaml")
    parser.add_argument('-t', '--modelype', help='type of model to run', required=False, default="MLP")
    parser.add_argument('-e', '--export_onnx', help='export to onnx format', required=False, action="store_true")
    args = parser.parse_args()
    return vars(args)

def remove_speciale_chars(str, chars = [",", ":", "-", " ", "."]):
    for char in chars:
        str = str.replace(char, "_")
    return str


lambda_chan = (lambda x: x.type(torch.int8))

def load_dataset(config):
    if config['dataset'] == 'mnist':
#        import ipdb as pdb; pdb.set_trace()
        # MNIST Dataset
        train_trans = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Lambda(lambda_chan),
                    ]
              )
        train_dataset = dsets.MNIST(root='./data/',
                                    train=True, 
                                    transform=train_trans, #transforms.ToTensor()
                                    download=True)

        test_dataset = dsets.MNIST(root='./data/',
                                   train=False, 
                                   transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config["batchsize"], 
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config["batchsize"], 
                                                  shuffle=False)
    else: 
        print ("Unsupported dataset type".format(config['dataset']))
        sys.exit()
    return train_loader, test_loader, train_dataset, test_dataset

def save_model(model, config):
    if not ("experiment_name" in config):
        experiment_name =  None
    elif config["experiment_name"] == "":
        experiment_name = None
    else:
        experiment_name = config["experiment_name"]

    current_datetime = str(datetime.datetime.now())
    current_datetime = remove_speciale_chars(current_datetime)
    if experiment_name == None:
        experiment_name = current_datetime
    else:
        if os.path.exists(experiment_name+".pkl"):
            experiment_name += "_"+current_datetime
    experiment_name += ".pkl"
    torch.save(model.state_dict(), experiment_name)



