import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utility
import config
import models 

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def build_model(config):
    model_type = config['model_type']
    # import ipdb as pdb; pdb.set_trace()
    if model_type =='lenet':
        model = models.LENET(config)
    elif model_type == 'mlp':
        model = models.MLP(config)
    else:
        print("model_type={0} is not supported yet!".fortmat(model_type))
    if config['operation_mode'] == "retrain" or config['operation_mode'] == "inference":
        print("Using a trained model...")
        model.load_state_dict(torch.load(config['trained_model']))
    else:
        # Loss and Optimizer
        model.weights_init(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)
    return model, criterion, optimizer, scheduler

def test_model(test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images)
        # import ipdb as pdb; pdb.set_trace()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_accuracy = 100.0 * float(correct) / total
    return test_accuracy

def model_inference(test_loader, config):
    num_epochs = config['num_epochs']
    for epoch in range(num_epochs):
        test_accuracy = test_model(test_loader)
        print('Test Accuracy of the model on the 10000 test images: {0}'.format(test_accuracy))

def train_model(model, criterion, optimizer, scheduler, train_loader, train_dataset, test_loader, config):
    # Train the Model
    num_epochs = config['num_epochs']
    batch_size = config['batchsize']
    model_type = config['model_type']
    top_test_acc = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_accuracy = test_model(test_loader)
        scheduler.step()
#        import ipdb as pdb; pdb.set_trace()
        print('[{0}] Test Accuracy of the model on the 10000 test images: {1} , lr:{2}, loss:{3}'.format(epoch, test_accuracy, get_lr(optimizer), float(loss.data.cpu().numpy())))
        if test_accuracy > top_test_acc :
            utility.save_model(config=config, model=model)
            top_test_acc = test_accuracy
        # print('Test Accuracy of the model on the 10000 test images: {0}'.format(test_accuracy))

if __name__ == '__main__':
    args = utility.parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    model, criterion, optimizer, scheduler = build_model(config)
    # import ipdb as pdb; pdb.set_trace()
    if torch.cuda.is_available():
        model = model.cuda()
    train_loader, test_loader, train_dataset, test_dataset = utility.load_dataset(config)
    if config['operation_mode'] == "inference":
        model_inference(test_loader, config)
    else:
        train_model(model, criterion, optimizer, scheduler, train_loader, train_dataset, test_loader, config)
    # test_model(test_loader)
    # Save the Trained Model
    # import ipdb as pdb; pdb.set_trace()
