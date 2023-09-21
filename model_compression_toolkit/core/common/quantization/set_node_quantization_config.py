# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import copy
from typing import List

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeActivationQuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn
from model_compression_toolkit.core.common.quantization.quantization_fn_selection import \
    get_weights_quantization_fn
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.target_platform.op_quantization_config import OpQuantizationConfig, \
    QuantizationConfigOptions


def set_quantization_configuration_to_graph(graph: Graph,
                                            quant_config: QuantizationConfig,
                                            mixed_precision_enable: bool = False) -> Graph:
    """
    Add quantization configuration for each graph node.

    Args:
        graph: Graph for which to add quantization info to each node.
        quant_config: Quantization configuration containing parameters for how the graph should be quantized.
        mixed_precision_enable: is mixed precision enabled

    Returns:
        The graph with quantization configurations attached to each node in it.
    """
    first_node_name = ''
    last_node_name = ''
    
    for n in graph.nodes:
        node_qc_options = n.get_qco(graph.tpc)
        if node_qc_options.base_config.enable_activation_quantization :
            first_node_name = n.name
            break
    for n in graph.nodes:
        node_qc_options = n.get_qco(graph.tpc)
        if node_qc_options.base_config.enable_activation_quantization:
            last_node_name = n.name
    first_and_last_node_to_promote = [first_node_name, last_node_name]
    for n in graph.nodes:
        set_quantization_configs_to_node(node=n,
                                         quant_config=quant_config,
                                         fw_info=graph.fw_info,
                                         tpc=graph.tpc,
                                         mixed_precision_enable=mixed_precision_enable,
                                         first_and_last_node_to_promote=first_and_last_node_to_promote)
    return graph


def set_quantization_configs_to_node(node: BaseNode,
                                     quant_config: QuantizationConfig,
                                     fw_info: FrameworkInfo,
                                     tpc: TargetPlatformCapabilities,
                                     mixed_precision_enable: bool = False,
                                     first_and_last_node_to_promote: list = [None, None]):
    """
    Create and set quantization configurations to a node (for both weights and activation).

    Args:
        node: Node to set its quantization configurations.
        quant_config: Quantization configuration to generate the node's configurations from.
        fw_info: Information needed for quantization about the specific framework.
        tpc: TargetPlatformCapabilities to get default OpQuantizationConfig.
        mixed_precision_enable: is mixed precision enabled
    """
    node_qc_options = node.get_qco(tpc)
    layers_to_promote = node_qc_options.base_config.layers_to_promote
    original_bits = node_qc_options.base_config.activation_n_bits
    if 'first_and_last' in layers_to_promote.keys():
        bit_width = layers_to_promote['first_and_last']
        if node.name in first_and_last_node_to_promote:
            node_qc_options.base_config.activation_n_bits = bit_width
            print('PROMOTING: ', node.name)
    if node.name in layers_to_promote.keys():
        with open('/home/shreyas.kera/ng50_promote_layerwise_MSE.txt', 'a') as f:
            f.writelines('PROMOTING: '+str(node.name)+'\n')
        print('PROMOTING: ', node.name)
        node_qc_options.base_config.activation_n_bits = layers_to_promote[node.name]
    # print(node.name, 'ACTIVATION BITS', node_qc_options.base_config.activation_n_bits)
    # Create QC candidates for weights and activation combined
    weight_channel_axis = fw_info.kernel_channels_mapping.get(node.type)[0]
    
    node.candidates_quantization_cfg = _create_node_candidates_qc(quant_config,
                                                                fw_info,
                                                                weight_channel_axis,
                                                                node_qc_options,
                                                                mixed_precision_enable=mixed_precision_enable)

    for candidate_qc in node.candidates_quantization_cfg:
        candidate_qc.weights_quantization_cfg.enable_weights_quantization = \
            candidate_qc.weights_quantization_cfg.enable_weights_quantization and node.has_weights_to_quantize(fw_info)
        candidate_qc.activation_quantization_cfg.enable_activation_quantization = \
            candidate_qc.activation_quantization_cfg.enable_activation_quantization and node.get_has_activation()
    # for qc in node.candidates_quantization_cfg:
    #     if node.name not in node_qc_options.base_config.layers_to_not_quantize:
    #         qc.activation_quantization_cfg.enable_activation_quantization = False
    #         qc.weights_quantization_cfg.enable_weights_quantization = False
    #     else:
    #         pass
    #         with open('/home/shreyas.kera/ng50_demote_layerwise_MSE.txt', 'a') as f:
    #             f.writelines('DEMOTING '+node.name+'\n')
    node_qc_options.base_config.activation_n_bits = original_bits


def create_node_activation_qc(qc: QuantizationConfig,
                              fw_info: FrameworkInfo,
                              op_cfg: OpQuantizationConfig) -> NodeActivationQuantizationConfig:
    """
    Create an activation quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its
        weights/activations should be quantized)
        op_cfg: OpQuantizationConfig with quantizers types to set in node quantization configuration.

    Returns:
        Activation quantization configuration of a node.
    """

    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown quantization method for activations')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    return NodeActivationQuantizationConfig(qc,
                                            op_cfg,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def create_node_qc_candidate(qc: QuantizationConfig,
                             fw_info: FrameworkInfo,
                             weight_channel_axis: int,
                             op_cfg: OpQuantizationConfig) -> CandidateNodeQuantizationConfig:
    """
    Create quantization configuration candidate from a QuantizationConfig object.
    Creates both weights and activation quantization configurations
    and initialize a candidate object that encapsulates both.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether its
            weights/activations should be quantized)
        weight_channel_axis: Output channel index of the node's kernel.
        op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.

    Returns: a CandidateNodeQuantizationConfig object with both weights and activation quantization config objects.

    """

    # get attributes for weights quantization
    weights_quantization_fn = get_weights_quantization_fn(op_cfg.weights_quantization_method)

    if weights_quantization_fn is None:
        Logger.critical('Unknown quantization method for weights')  # pragma: no cover

    weights_quantization_params_fn = get_weights_quantization_params_fn(op_cfg.weights_quantization_method)

    # get attributes for activation quantization
    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(op_cfg.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown quantization method for activations')  # pragma: no cover

    activation_quantization_params_fn = get_activation_quantization_params_fn(op_cfg.activation_quantization_method)

    return CandidateNodeQuantizationConfig(qc=qc,
                                           op_cfg=op_cfg,
                                           activation_quantization_fn=activation_quantization_fn,
                                           activation_quantization_params_fn=activation_quantization_params_fn,
                                           weights_quantization_fn=weights_quantization_fn,
                                           weights_quantization_params_fn=weights_quantization_params_fn,
                                           weight_channel_axis=weight_channel_axis)


def _create_node_candidates_qc(qc: QuantizationConfig,
                               fw_info: FrameworkInfo,
                               weight_channel_axis: int,
                               node_qc_options: QuantizationConfigOptions,
                               mixed_precision_enable: bool = False) -> List[CandidateNodeQuantizationConfig]:
    """
    Create a list of candidates of weights and activation quantization configurations for a node.

    Args:
        qc: Quantization configuration the quantization process should follow.
        fw_info: Framework information (e.g., which layers should have their kernels' quantized).
        weight_channel_axis: Output channel index of the node's kernel.
        node_qc_options: QuantizationConfigOptions for the node with quantization candidates information.
        mixed_precision_enable: is mixed precision enabled

    Returns:
        List of candidates of weights quantization configurations to set for a node.
    """

    candidates = []
    if mixed_precision_enable:
        for op_cfg in node_qc_options.quantization_config_list:
            candidate_nbits_qc = copy.deepcopy(qc)
            candidates.append(create_node_qc_candidate(candidate_nbits_qc,
                                                       fw_info,
                                                       weight_channel_axis,
                                                       op_cfg))
        # sorting the candidates by weights number of bits first and then by activation number of bits
        # (in reversed order)
        candidates.sort(key=lambda c: (c.weights_quantization_cfg.weights_n_bits,
                                       c.activation_quantization_cfg.activation_n_bits), reverse=True)
    else:
        candidates.append(create_node_qc_candidate(qc,
                                                   fw_info,
                                                   weight_channel_axis,
                                                   node_qc_options.base_config))

    return candidates
