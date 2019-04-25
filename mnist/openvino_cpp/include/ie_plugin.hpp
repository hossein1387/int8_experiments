// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for Main Inference Engine API
 * @file ie_plugin.hpp
 */
#pragma once

#include <ie_icnn_network.hpp>
#include <ie_iextension.h>
#include "ie_api.h"
#include "details/ie_no_copy.hpp"
#include "ie_error.hpp"
#include "ie_version.hpp"
#include "ie_iexecutable_network.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>


#if defined(_WIN32)
    #ifdef IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        #define INFERENCE_PLUGIN_API(type) extern "C"   __declspec(dllexport) type
    #else
        #define INFERENCE_PLUGIN_API(type) extern "C" type
    #endif
#elif(__GNUC__ >= 4)
    #ifdef IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        #define INFERENCE_PLUGIN_API(type) extern "C"   __attribute__((visibility("default"))) type
    #else
        #define INFERENCE_PLUGIN_API(type) extern "C" type
    #endif
#else
    #define INFERENCE_PLUGIN_API(TYPE) extern "C" TYPE
#endif

namespace InferenceEngine {

/**
 * @brief Responce structure encapsulating information about supported layer
 */
struct QueryNetworkResult {
    std::set<std::string> supportedLayers;
    StatusCode rc;
    ResponseDesc resp;
};

/**
 * @brief This class is a main plugin interface
 */
class IInferencePlugin : public details::IRelease {
public:
    /**
     * @brief Returns plugin version information
     * @param versionInfo Pointer to version info. Is set by plugin
     */
    virtual void GetVersion(const Version *&versionInfo) noexcept = 0;

    /**
     * @brief Sets logging callback
     * Logging is used to track what is going on inside
     * @param listener Logging sink
     */
    virtual void SetLogCallback(IErrorListener &listener) noexcept = 0;

    /**
     * @deprecated use LoadNetwork with four parameters (executable network, cnn network, config, response)
     * @brief Loads a pre-built network with weights to the engine. In case of success the plugin will
     *        be ready to infer
     * @param network Network object acquired from CNNNetReader
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode LoadNetwork(ICNNNetwork &network, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Creates an executable network from a network object. User can create as many networks as they need and use
     *        them simultaneously (up to the limitation of the hardware resources)
     * @param ret Reference to a shared ptr of the returned network interface
     * @param network Network object acquired from CNNNetReader
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load operation
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode
    LoadNetwork(IExecutableNetwork::Ptr &ret, ICNNNetwork &network, const std::map<std::string, std::string> &config,
                ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Creates an executable network from a previously exported network
     * @param ret Reference to a shared ptr of the returned network interface
     * @param modelFileName Path to the location of the exported file
     * @param config Map of pairs: (config parameter name, config parameter value) relevant only for this load operation*
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode
    ImportNetwork(IExecutableNetwork::Ptr &ret, const std::string &modelFileName,
                  const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept = 0;

    /**
     * @deprecated Uses Infer() working with multiple inputs and outputs
     * @brief Infers an image(s).
     * Input and output dimensions depend on the topology.
     *     As an example for classification topologies use a 4D Blob as input (batch, channels, width,
     *             height) and get a 1D blob as output (scoring probability vector). To Infer a batch,
     *             use a 4D Blob as input and get a 2D blob as output in both cases the method will
     *             allocate the resulted blob
     * @param input Any TBlob<> object that contains the data to infer. The type of TBlob must match the network input precision and size.
     * @param result Related TBlob<> object that contains the result of the inference action, typically this is a float blob.
               The blob does not need to be allocated or initialized, the engine allocates the relevant data.
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode Infer(const Blob &input, Blob &result, ResponseDesc *resp) noexcept = 0;

    /**
     * @deprecated Loads IExecutableNetwork to create IInferRequest.
     * @brief Infers tensors. Input and output dimensions depend on the topology.
     *     As an example for classification topologies use a 4D Blob as input (batch, channels, width,
     *             height) and get a 1D blob as output (scoring probability vector). To Infer a batch,
     *             use a 4D Blob as input and get a 2D blob as output in both cases the method will
     *             allocate the resulted blob
     * @param input Map of input blobs accessed by input names
     * @param result Map of output blobs accessed by output names
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode Infer(const BlobMap &input, BlobMap &result, ResponseDesc *resp) noexcept = 0;

    /**
     * @deprecated Uses IInferRequest to get performance measures
     * @brief Queries performance measures per layer to get feedback of what is the most time consuming layer
     *  Note: not all plugins provide meaningful data
     * @param perfMap Map of layer names to profiling information for that layer
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo> &perfMap,
                                            ResponseDesc *resp) const noexcept = 0;

    /**
     * @brief Registers extension within the plugin
     * @param extension Pointer to already loaded extension
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     * @return Status code of the operation. OK if succeeded
     */
    virtual StatusCode AddExtension(InferenceEngine::IExtensionPtr extension,
                                    InferenceEngine::ResponseDesc *resp) noexcept = 0;

    /**
     * @brief Sets configuration for plugin, acceptable keys can be found in ie_plugin_config.hpp
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    virtual StatusCode SetConfig(const std::map<std::string, std::string> &config, ResponseDesc *resp) noexcept = 0;


    /**
     * @depricated Use the version with config parameter
     * @brief Query plugin if it supports specified network
     * @param network Network object to query
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    virtual void QueryNetwork(const ICNNNetwork& /*network*/, QueryNetworkResult& res) const noexcept {
        res.rc = InferenceEngine::NOT_IMPLEMENTED;
    }

    /**
     * @brief Query plugin if it supports specified network with specified configuration
     * @param network Network object to query
     * @param config Map of pairs: (config parameter name, config parameter value)
     * @param resp Pointer to the response message that holds a description of an error if any occurred
     */
    virtual void QueryNetwork(const ICNNNetwork& /*network*/,
                              const std::map<std::string, std::string> &/*config*/, QueryNetworkResult& res) const noexcept {
        res.rc = InferenceEngine::NOT_IMPLEMENTED;
    }
};

/**
 * @brief Creates the default instance of the interface (per plugin)
 * @param plugin Pointer to the plugin
 * @param resp Pointer to the response message that holds a description of an error if any occurred
 * @return Status code of the operation. OK if succeeded
 */
INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin *&plugin, ResponseDesc *resp) noexcept;
}  // namespace InferenceEngine
