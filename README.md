# Using OpenVino to run a Pytorch model on Intel CPU


# First train your model:
    cd $int8_experiment/pytorch
    python main.py -f config.yaml -t LENET
    ========================================================
    Configuration:
    ========================================================
    using pytorch: 1.0.1.post2
    dataset: mnist
    lr: 0.001
    batchsize: 100
    num_epochs: 100
    model_type: lenet
    init_type: glorot
    quantization: normal
    operation_mode: normal
    experiment_name: mnist
    trained_model: ./mnist.pkl

    [0] Test Accuracy of the model on the 10000 test images: 96.63 , lr:0.001    , loss:0.206096425
    [1] Test Accuracy of the model on the 10000 test images: 97.3  , lr:0.00099  , loss:0.014005098
    [2] Test Accuracy of the model on the 10000 test images: 97.96 , lr:0.0009801, loss:0.097614221
    ...

# Run Openvino optimzer and inference engine:
    cd $int8_experiment/openvino_py
    python openvino_mnist.py -f config.yaml -t LENET
    ========================================================
    Configuration:
    ========================================================
    using pytorch: 1.0.1.post2
    dataset: mnist
    lr: 0.001
    batchsize: 100
    num_epochs: 1
    model_type: lenet
    init_type: glorot
    quantization: normal
    operation_mode: normal
    experiment_name: mnist
    trained_model: ./mnist.pkl

    [INFO   ]   =======================================================================
    [INFO   ]   exporting ./mnist.pkl to ONNX
    [INFO   ]   =======================================================================
    mnist.onnx exported!
    [INFO   ]   =======================================================================
    [INFO   ]   Running OpenVino optimizer on mnist.onnx
    [INFO   ]   =======================================================================
    Model Optimizer arguments:
    Common parameters:
        - Path to the Input Model:  /home/mhossein/myRepos/int8_experiment/openvino_py/mnist.onnx
        - Path for generated IR:    /home/mhossein/myRepos/int8_experiment/openvino_py/.
        - IR output name:   mnist
        - Log level:    ERROR
        - Batch:    Not specified, inherited from the model
        - Input layers:     Not specified, inherited from the model
        - Output layers:    Not specified, inherited from the model
        - Input shapes:     Not specified, inherited from the model
        - Mean values:  Not specified
        - Scale values:     Not specified
        - Scale factor:     Not specified
        - Precision of IR:  FP32
        - Enable fusing:    True
        - Enable grouped convolutions fusing:   True
        - Move mean values to preprocess section:   False
        - Reverse input channels:   False
    ONNX specific parameters:
    Model Optimizer version:    1.5.12.49d067a0

    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/mhossein/myRepos/int8_experiment/openvino_py/./mnist.xml
    [ SUCCESS ] BIN file: /home/mhossein/myRepos/int8_experiment/openvino_py/./mnist.bin
    [ SUCCESS ] Total execution time: 0.42 seconds. 
    [INFO   ]   =======================================================================
    [INFO   ]   Running Openvino Inference on 10000 images
    [INFO   ]   =======================================================================
    name                                                                   layer_type      exet_type       status          real_time, us
    28                                                                     Convolution     jit_avx2_FP32   EXECUTED        351       
    30                                                                     Pooling         jit_avx_FP32    EXECUTED        19        
    31                                                                     Convolution     ref_any_FP32    EXECUTED        5828      
    33                                                                     Pooling         jit_avx_FP32    EXECUTED        742       
    34                                                                     Convolution     jit_avx2_FP32   EXECUTED        167       
    34_nChw8c_nchw_43                                                      Reorder         reorder_FP32    EXECUTED        21        
    43                                                                     Reshape         unknown_FP32    NOT_RUN         0         
    44                                                                     FullyConnected  jit_gemm_FP32   EXECUTED        207       
    45                                                                     FullyConnected  jit_gemm_FP32   EXECUTED        18        
    46                                                                     FullyConnected  jit_gemm_FP32   EXECUTED        8         
    out_46                                                                 Output          unknown_FP32    NOT_RUN         0         
    accuracy = 0.8979


# Quantize model for int8:

    ~/inference_engine_samples_build/intel64/Release/calibration_tool -t C -d CPU -i ./mnist_dataset/mnist_data -m mnist.xml -threshold 10
    [ INFO ] InferenceEngine: 
        API version ............ 1.4
        Build .................. 19154
    [ INFO ] Parsing input parameters
    [ INFO ] Loading plugin

        API version ............ 1.5
        Build .................. lnx_20181004
        Description ....... MKLDNNPlugin
    [ INFO ] Loading network files
    [ INFO ] Preparing input blobs
    [ INFO ] Batch size is 1
    [ INFO ] Collecting accuracy metric in FP32 mode to get a baseline, collecting activation statistics
    Progress: [....................] 100.00% done
      FP32 Accuracy: 20.17% 
    [ INFO ] Verification of network accuracy if all possible layers converted to INT8
    Validate int8 accuracy, threshold for activation statistics = 100.00
    Progress: [....................] 100.00% done
       Accuracy is 12.50%
    Validate int8 accuracy, threshold for activation statistics = 99.50
    Progress: [....................] 100.00% done
       Accuracy is 12.83%
    Validate int8 accuracy, threshold for activation statistics = 99.00
    Progress: [....................] 100.00% done
       Accuracy is 13.33%
    Validate int8 accuracy, threshold for activation statistics = 98.50
    Progress: [....................] 100.00% done
       Accuracy is 12.00%
    Validate int8 accuracy, threshold for activation statistics = 98.00
    Progress: [....................] 100.00% done
       Accuracy is 12.67%
    Validate int8 accuracy, threshold for activation statistics = 97.50
    Progress: [....................] 100.00% done
       Accuracy is 13.17%
    Validate int8 accuracy, threshold for activation statistics = 97.00
    Progress: [....................] 100.00% done
       Accuracy is 12.50%
    Validate int8 accuracy, threshold for activation statistics = 96.50
    Progress: [....................] 100.00% done
       Accuracy is 12.17%
    Validate int8 accuracy, threshold for activation statistics = 96.00
    Progress: [....................] 100.00% done
       Accuracy is 12.50%
    Validate int8 accuracy, threshold for activation statistics = 95.50
    Progress: [....................] 100.00% done
       Accuracy is 12.67%
    [ INFO ] Achieved required accuracy drop satisfying threshold
    FP32 accuracy: 20.17% vs current Int8 configuration accuracy: 13.33% with threshold for activation statistic: 99.00%
    Layers profile for Int8 quantization
    28: I8
    31: I8
    34: I8
