import argparse
import numpy as np
import onnxruntime as rt
import torch
import torchvision
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--onnx_model', help='input onnx model', required=True)
    args = parser.parse_args()
    return vars(args)

def get_images(num_images):
    images = np.ndarray(shape=(num_images, 1, 28, 28))
    labels = np.ndarray(shape=(num_images))
    BATCH_SIZE = 1
    # transform = transforms.ToTensor()
    # lambda_shift = (lambda x: 255*x-128)
    transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
    trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    dataiter = iter(trainloader)
    for i in range(0, num_images):
        image, label = dataiter.next()
        images[i,:,:,:] = image[0,:,:]
        labels[i]       = label
    return images, labels

def load_images(num_images):
    images, labels = get_images(num_images)
    return images, labels

def do_inference(onnx_model):
    num_tests  = 100
    num_images = 100
    # import ipdb as pdb; pdb.set_trace()
    sess = rt.InferenceSession(onnx_model)
    input_name = sess.get_inputs()[0].name
    pass_cnt = 0
    for test_id in range(0, num_tests):
        images, labels = load_images(num_images)
        pred_onx = sess.run(None, {input_name: images.astype(np.float32)})[0]
        for i in range(0, num_images):
            pred = pred_onx[i].argmax()
            res = "FAIL"
            if labels[i] == pred:
                res = "PASS"
                pass_cnt += 1
            print("actual={}   pred={}   res={}".format(labels[i], pred, res))
    print("Accuracy = {}% out of {} tests".format(100*pass_cnt/(float(num_images)*num_tests), float(num_images)*num_tests))


if __name__ == '__main__':
    args = parse_args()
    onnx_model = args['onnx_model']
    do_inference(onnx_model)
