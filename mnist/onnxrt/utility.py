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
import os
import sys
import subprocess
from termcolor import colored
import platform

verbose = {"VERB_NONE":0, "VERB_LOW":100, "VERB_MEDIUM":200,"VERB_HIGH":300, "VERB_FULL":400, "VERB_DEBUG":500}

def run_command(command_str, split=True, verbosity="VERB_HIGH"):
        try:
            print_log(command_str, id_str="command", verbosity=verbosity)
            # subprocess needs to receive args seperately
            if split:
                res = subprocess.call(command_str.split())
            else:
                res = subprocess.call(command_str, shell=True)
            if res == 1:
                print_log("Errors while executing: {0}".format(command_str), "ERROR", verbosity="VERB_LOW")
                sys.exit()
        except OSError as e:
            print_log("Unable to run {0} command".format(command_str), "ERROR")
            sys.exit()

def print_log(log_str, id_str="INFO", color="white", verbosity="VERB_LOW"):
    if verbosity not in verbose:
        print_log("Unknown verbosity {0} choose from {1}".format(verbosity, verbose.keys()), "ERROR")
        sys.exit()
    if verbose[verbosity] < verbose["VERB_MEDIUM"]:
        if "white" in color.lower():
            if "warning" in id_str.lower():
                color = "yellow"
            elif "error" in id_str.lower():
                color = "red"
            elif "command" in id_str.lower():
                color = "green"
            elif "pass" in id_str.lower():
                color = "green"
        print(colored(("[{0:<7}]   {1}".format(id_str, log_str)), color))

def print_banner(banner_str, color="white", verbosity="VERB_LOW"):
    print_log("=======================================================================", color=color, verbosity=verbosity)
    print_log(banner_str, color=color, verbosity=verbosity)
    print_log("=======================================================================", color=color, verbosity=verbosity)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False, default="config.yaml")
    parser.add_argument('-t', '--modelype', help='type of model to run', required=False, default="MLP")
    args = parser.parse_args()
    return vars(args)

def remove_speciale_chars(str, chars = [",", ":", "-", " ", "."]):
    for char in chars:
        str = str.replace(char, "_")
    return str


# lambda_chan = (lambda x: x.type(torch.int8))

def load_dataset(config):
    if config['dataset'] == 'mnist':
#        import ipdb as pdb; pdb.set_trace()
        # MNIST Dataset
        # train_trans = transforms.Compose(
        #             [
        #                 transforms.ToTensor(),
        #                 transforms.Lambda(lambda_chan),
        #             ]
        #       )
        train_dataset = dsets.MNIST(root='./data/',
                                    train=True, 
                                    transform=transforms.ToTensor(),#train_trans
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
    print("saving model to {}".format(experiment_name))
    torch.save(model.state_dict(), experiment_name)
    export_torch_to_onnx(model, (100, 1, 28, 28))

def export_torch_to_onnx(model, shape):
    batch_size, nb_channels, w, h = shape
    import torch
    #import ipdb as pdb; pdb.set_trace()
    if isinstance(model, torch.nn.Module):
        model_name =  model.__class__.__name__
        # create the imput placeholder for the model
        # note: we have to specify the size of a batch of input images
        input_placeholder = torch.randn(batch_size, nb_channels, w, h)
        onnx_model_fname = model_name + ".onnx"
        # export pytorch model to onnx
        torch.onnx.export(model, input_placeholder, onnx_model_fname)
        print("{0} was exported to onnx: {1}".format(model_name, onnx_model_fname))
        return onnx_model_fname
    else:
        print("Unsupported model file")
        return

