import onnxruntime as rt
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--input_model', help='input onnx model', required=True)
    parser.add_argument('-d', '--dataset_path', help='path to input dataset', required=True)
    args = parser.parse_args()
    return vars(args)

def load_input(dataset_path):
    #import ipdb as pdb; pdb.set_trace()
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    transform_test = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataloader = datasets.CIFAR10
    testset = dataloader(root=dataset_path, train=False, download=True, transform=transform_test)
    return testset

def main(input_model, dataset_path):
    # import ipdb as pdb; pdb.set_trace()
    sess = rt.InferenceSession(input_model)
    input_name = sess.get_inputs()[0].name
    dloader = load_input(dataset_path)
    pass_cnt = 0
    cnt      = 0
    for batch in dloader:
        image = batch[0]
        label = batch[1]
        image = np.expand_dims(image, axis=1)
        image = np.moveaxis(image, 0, 1)
        pred_onnx = sess.run(None, {input_name: image})
        cnt += 1
        if pred_onnx[0][0].argmax() == label:
            pass_cnt += 1
        if cnt > 100:
            break
    print("\n\n\nAccuracy: {0}".format(pass_cnt/float(cnt)))

if __name__=='__main__':
    args = parse_args()
    input_model  = args['input_model']
    dataset_path = args['dataset_path']
    main(input_model, dataset_path)
