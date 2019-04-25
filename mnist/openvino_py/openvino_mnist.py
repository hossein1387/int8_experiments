import argparse
import torch
import torch.onnx
import onnx
import models 
import utility as util
import config
from openvino.inference_engine import IENetwork, IEPlugin
import numpy as np
test_dataset = 'data/mnist_test_data.npz'
openvino_inst_path = "/opt/intel/openvino"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--configfile', help='config file in yaml format', required=False, default="config.yaml")
    parser.add_argument('-t', '--modelype', help='type of model to run', required=False, default="MLP")
    parser.add_argument('-w', '--weights', help='binary file containing weights', required=False)
    parser.add_argument('-m', '--model', help='xml file containing model', required=False)
    args = parser.parse_args()
    return vars(args)

def export_to_onnx(config):
    util.print_banner("exporting {0} to ONNX".format(config['trained_model']), color='green', verbosity="VERB_LOW")
    # import ipdb as pdb; pdb.set_trace()
  # model object
    model = models.LENET(config)
    # load weights
    trained_model = torch.load(config['trained_model'])
    model.load_state_dict(trained_model)
    # create the imput placeholder for the model
    # note: we have to specify the size of a batch of input images
    input_placeholder = torch.randn(1, 1, 28, 28)
    # export
    onnx_model_fname = config['experiment_name'] + ".onnx"
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
    plugin = IEPlugin(device=device, plugin_dirs=None)
    net = IENetwork(model=model_xml, weights=model_bin)
    exec_net = plugin.load(network=net)
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    return exec_net, input_blob, output_blob

def load_input(datafile):
    f = np.load(datafile)
    return f['images'], f['labels']

def print_perf_counts(exec_net):
    perf_counts = exec_net.requests[0].get_perf_counts()
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
    for layer, stats in perf_counts.items():
      print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'], stats['status'], stats['real_time']))

def run_inference(exec_net, input_blob, output_blob, images, labels):
    # import ipdb as pdb; pdb.set_trace()
    util.print_banner("Running Openvino Inference on {0} images".format(images.shape[0]), color='green', verbosity="VERB_LOW")
    data_counts = images.shape[0]
    hit_counts = 0
    for i in range(data_counts):
      res = exec_net.infer(inputs={input_blob: images[i]})
      pred = res[output_blob].argmax()
      if pred == labels[i]:
        hit_counts += 1 
    accuracy = float(hit_counts)/float(data_counts)
    print_perf_counts(exec_net)
    return accuracy

def optimize_model(expr_name, onnx_file, model_xml, weight_bin):
    run_opt = False
    if (model_xml == None):
        util.print_log("Could not find xml model", id_str="warning")
        run_opt = True
    if (weight_bin == None):
        util.print_log("Could not find binary weights", id_str="warning")
        run_opt = True
    if run_opt:
        util.print_banner("Running OpenVino optimizer on {0}".format(onnx_file), color='green', verbosity="VERB_LOW")
        cmd = "python {0}/deployment_tools/model_optimizer/mo.py --input_model={1} --model_name {2}".format(openvino_inst_path, onnx_file, expr_name)
        util.run_command(cmd, verbosity="VERB_LOW")
        model_xml, weight_bin = expr_name+".xml", expr_name+".bin"
    # load model
    # import ipdb as pdb; pdb.set_trace()
    return model_xml, weight_bin

def main(config, model_xml, weight_bin):
    onnx_file = export_to_onnx(config) 
    model_xml, weight_bin = optimize_model(config['experiment_name'], onnx_file, model_xml, weight_bin)
    exec_net, input_blob, output_blob = load_model("CPU", model_xml, weight_bin)
    # load input
    images, labels = load_input(test_dataset)
    # run inference
    accuracy = run_inference(exec_net, input_blob, output_blob, images, labels)
    print('accuracy = {}'.format(accuracy))

if __name__=='__main__':
    args = parse_args()
    model_type = args['modelype']
    config_file = args['configfile']
    model_xml = args['model']
    weight_bin = args['weights']
    config = config.Configuration(model_type, config_file)
    print(config.get_config_str())
    config = config.config_dict
    main(config, model_xml, weight_bin)

