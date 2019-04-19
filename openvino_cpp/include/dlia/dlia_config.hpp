// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_plugin_config.hpp
 */

#pragma once

#include <string>
#include "../ie_plugin_config.hpp"

namespace InferenceEngine {

namespace DLIAConfigParams {

#define DLIA_CONFIG_KEY(name) InferenceEngine::DLIAConfigParams::_CONFIG_KEY(DLIA_##name)
#define DECLARE_DLIA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(DLIA_##name)
#define DECLARE_DLIA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(DLIA_##name)

/**
 * @brief The key to define the type of transformations for DLIA inputs and outputs.
 * DLIA use custom data layout for input and output blobs. IE DLIA Plugin provides custom
 * optimized version of transformation functions that do not use OpenMP and much more faster
 * than native DLIA functions. Values: "DLIA_IO_OPTIMIZED" - optimized plugin transformations
 * are used, "DLIA_IO_NATIVE" - native DLIA transformations are used.
 */
DECLARE_DLIA_CONFIG_KEY(IO_TRANSFORMATIONS);

DECLARE_DLIA_CONFIG_VALUE(IO_OPTIMIZED);
DECLARE_DLIA_CONFIG_VALUE(IO_NATIVE);

DECLARE_DLIA_CONFIG_KEY(ARCH_ROOT_DIR);
DECLARE_DLIA_CONFIG_KEY(PERF_ESTIMATION);

}  // namespace DLIAConfigParams
}  // namespace InferenceEngine
