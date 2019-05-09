import argparse
import torch
import torch.onnx
import onnx
import utility as util
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
import models.cifar as models


#test_dataset = 'data/mnist_test_data.npz'
openvino_inst_path = "/opt/intel/openvino"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', help='binary file containing weights', required=False)
    parser.add_argument('-m', '--model'  , help='xml file containing model', required=False)
    parser.add_argument('-a', '--arch'   , metavar='ARCH', default='resnet20', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('-c', '--chkpoint'  , help='checkpoint', required=True)
    parser.add_argument('-d', '--dataset', help='path to the model dataset', required=True) 
    parser.add_argument('-o', '--opt_op', help="openvino optimization options", required=False, nargs='+')
    args = parser.parse_args()
    return vars(args)


def load_state_dict(new_model, trained_state_dict):
    #import ipdb as pdb; pdb.set_trace()
    new_state_dict = dict.fromkeys(new_model.state_dict().keys())
    for key in trained_state_dict.keys():
        new_key = str(key).split("module.")[1]
        new_state_dict[new_key] = trained_state_dict[key]
        #print("{0} is copied to {1}".format(key, new_key))
    new_model.load_state_dict(new_state_dict)
    #import ipdb as pdb; pdb.set_trace()
    return new_model

def export_to_onnx(args):
#    import ipdb as pdb; pdb.set_trace()
    util.print_banner("exporting {0} to ONNX".format(args['arch']), color='green', verbosity="VERB_LOW")
   
  # model object
    model = models.__dict__[args['arch']](num_classes=10)
    #model = torch.nn.DataParallel(model).cuda()
    # load weights
    checkpoint = torch.load(args['chkpoint'])
    model = load_state_dict(model, checkpoint['state_dict'])
    #model.load_state_dict(checkpoint['state_dict'])
    # create the imput placeholder for the model
    # note: we have to specify the size of a batch of input images
    input_placeholder = torch.randn(1, 3, 32, 32)
    # export
    onnx_model_fname = args['arch'] + ".onnx"
    #import ipdb as pdb; pdb.set_trace()
    torch.onnx.export(model, input_placeholder, onnx_model_fname)
    print('{} exported!'.format(onnx_model_fname))
    # print_onnx(onnx_model_fname)
    return onnx_model_fname

def print_onnx(onnx_model):
  model = onnx.load(onnx_model)
  onnx.checker.check_model(model)
  print('Contents of this model {}:'.format(onnx_model))
  print(onnx.helper.printable_graph(model.graph))


def load_model(device, model_xml, model_bin):
    #import ipdb as pdb; pdb.set_trace()
    plugin = IEPlugin(device=device)
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net)
    del net
    return exec_net, input_blob, output_blob

def save_data_as_image(data, name):
 #   import ipdb as pdb; pdb.set_trace()
    from imageio import imsave
    image = data[0]
    image = np.moveaxis(image, 0, -1)
    imsave(name, image)

def load_input(dataset_path):
    #import ipdb as pdb; pdb.set_trace()
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    transform_test = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataloader = datasets.CIFAR10
    testset = dataloader(root=dataset_path, train=False, download=False, transform=transform_test)
    images = testset.data
    targets= testset.targets
    images = np.expand_dims(images, axis=1)
    #images = np.moveaxis(images, -1, 2)
    return images, targets

def print_perf_counts(exec_net):
    perf_counts = exec_net.requests[0].get_perf_counts()
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
    for layer, stats in perf_counts.items():
      print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], stats['status'], stats['real_time']))

def run_inference(exec_net, input_blob, output_blob, images, labels):
    #import ipdb as pdb; pdb.set_trace()
    util.print_banner("Running Openvino Inference on {0} images".format(images.shape[0]), color='green', verbosity="VERB_LOW")
    data_counts = images.shape[0]
    hit_counts = 0
    for i in range(data_counts):
      res = exec_net.infer(inputs={input_blob: images[i]})
      pred = res[output_blob].argmax()
      if pred == labels[i]:
        hit_counts += 1 
      #else:
      #    print("pred={0}  actual={1}".format(pred, labels[i]))
      #save_data_as_image(images[i], "l_" + str(labels[i])+"p_"+str(pred)+".png")
    accuracy = float(hit_counts)/float(data_counts)
    print_perf_counts(exec_net)
    return accuracy

def optimize_model(expr_name, onnx_file, model_xml, weight_bin, opt_ops=""):
#    import ipdb as pdb; pdb.set_trace()
    run_opt = False
    options = ""
    if (model_xml == None):
        util.print_log("No xml model was provided", id_str="warning")
        run_opt = True
    if (weight_bin == None):
        util.print_log("No binary weight file was provided", id_str="warning")
        run_opt = True
    if run_opt:
        util.print_banner("Running OpenVino optimizer on {0}".format(onnx_file), color='green', verbosity="VERB_LOW")
        options += "--input_model={0} ".format(onnx_file)
        options += "--model_name {0} ".format(expr_name)
        options += (" --"+opt_ops[0]) if len(opt_ops)==1 else  "".join(" --"+e for e in opt_ops)
        cmd = "python {0}/deployment_tools/model_optimizer/mo_onnx.py {1}".format(openvino_inst_path, options)
        util.run_command(cmd, verbosity="VERB_LOW")
        model_xml, weight_bin = expr_name+".xml", expr_name+".bin"
    # load model
    # import ipdb as pdb; pdb.set_trace()
    return model_xml, weight_bin

def main(__args__):
    onnx_file = export_to_onnx(__args__)
    experiment_name = __args__['arch']
    model_xml       = __args__['model']
    weight_bin      = __args__['weight']
    dataset_path    = __args__['dataset']
    optimization_opt= "" if (__args__['opt_op'] == None) else __args__['opt_op']
    model_xml, weight_bin = optimize_model(experiment_name, onnx_file, model_xml, weight_bin, optimization_opt)
    exec_net, input_blob, output_blob = load_model("CPU", model_xml, weight_bin)
    # load input
    images, labels = load_input(dataset_path)
    # run inference
    accuracy = run_inference(exec_net, input_blob, output_blob, images, labels)
    print('accuracy = {}'.format(accuracy))

if __name__=='__main__':
    __args__ = parse_args()
    main(__args__)

