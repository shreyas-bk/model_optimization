# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Any

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.constants import THRESHOLD, RANGE_MIN, RANGE_MAX, SIGNED, CLUSTER_CENTERS, SCALE_PER_CHANNEL
from model_compression_toolkit.core.common.quantization.node_quantization_config import BaseNodeQuantizationConfig, \
    NodeWeightsQuantizationConfig, NodeActivationQuantizationConfig

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer
from mct_quantizers import constants as qi_keras_consts


def get_inferable_quantizer_kwargs(node_qc: BaseNodeQuantizationConfig,
                                   quantization_target: QuantizationTarget) -> Dict[str, Any]:
    """
    Get the quantization parameters for an inferable quantizer.
    Args:
        node_qc: The node quantization configuration of the node for which the quantizer is being created.
            Needs to match the specific quantization target.
        quantization_target: The target of the quantization (weights or activations).

    Returns:
        The quantization parameters as a dictionary.
    """

    if quantization_target == QuantizationTarget.Weights:
        if not isinstance(node_qc, NodeWeightsQuantizationConfig):
            Logger.error(f"Non-compatible node quantization config was given for quantization target Weights.")  # pragma: no cover

        quantization_method = node_qc.weights_quantization_method

        # Return the appropriate quantization parameters based on the quantization method
        if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                                   QuantizationMethod.SYMMETRIC]:
            return {qi_keras_consts.NUM_BITS: node_qc.weights_n_bits,
                    qi_keras_consts.THRESHOLD: list(node_qc.weights_quantization_params[THRESHOLD].flatten()),
                    qi_keras_consts.PER_CHANNEL: node_qc.weights_per_channel_threshold,
                    qi_keras_consts.CHANNEL_AXIS: node_qc.weights_channels_axis,
                    qi_keras_consts.INPUT_RANK: len(node_qc.weights_quantization_params[THRESHOLD].shape)}

        elif quantization_method in [QuantizationMethod.UNIFORM]:
            return {qi_keras_consts.NUM_BITS: node_qc.weights_n_bits,
                    qi_keras_consts.PER_CHANNEL: node_qc.weights_per_channel_threshold,
                    qi_keras_consts.MIN_RANGE: list(node_qc.weights_quantization_params[RANGE_MIN].flatten()),
                    qi_keras_consts.MAX_RANGE: list(node_qc.weights_quantization_params[RANGE_MAX].flatten()),
                    qi_keras_consts.CHANNEL_AXIS: node_qc.weights_channels_axis,
                    qi_keras_consts.INPUT_RANK: len(node_qc.weights_quantization_params[RANGE_MIN].shape)}

        elif quantization_method in [QuantizationMethod.LUT_SYM_QUANTIZER, QuantizationMethod.LUT_POT_QUANTIZER]:
            return {qi_keras_consts.NUM_BITS: node_qc.weights_n_bits,
                    qi_keras_consts.PER_CHANNEL: node_qc.weights_per_channel_threshold,
                    qi_keras_consts.CLUSTER_CENTERS: node_qc.weights_quantization_params[CLUSTER_CENTERS],
                    qi_keras_consts.THRESHOLD: list(node_qc.weights_quantization_params[SCALE_PER_CHANNEL].flatten()),
                    qi_keras_consts.CHANNEL_AXIS: node_qc.weights_channels_axis,
                    # TODO: how to pass multiplier nbits and eps for a specific node?
                    qi_keras_consts.INPUT_RANK: len(node_qc.weights_quantization_params[SCALE_PER_CHANNEL].shape)}

        else:
            Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover

    elif quantization_target == QuantizationTarget.Activation:
        if not isinstance(node_qc, NodeActivationQuantizationConfig):
            Logger.error(f"Non-compatible node quantization config was given for quantization target Activation.")  # pragma: no cover

        quantization_method = node_qc.activation_quantization_method

        # Return the appropriate quantization parameters based on the quantization method
        if quantization_method in [QuantizationMethod.POWER_OF_TWO,
                                   QuantizationMethod.SYMMETRIC]:
            return {qi_keras_consts.NUM_BITS: node_qc.activation_n_bits,
                    # In activation quantization is per-tensor only - thus we hold the threshold as a list with a len of 1
                    qi_keras_consts.THRESHOLD: [node_qc.activation_quantization_params[THRESHOLD]],
                    qi_keras_consts.SIGNED: node_qc.activation_quantization_params[SIGNED]}

        elif quantization_method in [QuantizationMethod.UNIFORM]:
            return {qi_keras_consts.NUM_BITS: node_qc.activation_n_bits,
                    # In activation quantization is per-tensor only - thus we hold the min/max as a list with a len of 1
                    qi_keras_consts.MIN_RANGE: [node_qc.activation_quantization_params[RANGE_MIN]],
                    qi_keras_consts.MAX_RANGE: [node_qc.activation_quantization_params[RANGE_MAX]]}

        elif quantization_method in [QuantizationMethod.LUT_POT_QUANTIZER]:
            return {qi_keras_consts.NUM_BITS: node_qc.activation_n_bits,
                    qi_keras_consts.SIGNED: node_qc.activation_quantization_params[SIGNED],
                    qi_keras_consts.CLUSTER_CENTERS: node_qc.activation_quantization_params[CLUSTER_CENTERS],
                    qi_keras_consts.THRESHOLD: [node_qc.activation_quantization_params[THRESHOLD]]
                    # TODO: how to pass multiplier nbits and eps for a specific node?
                    }
        else:
            Logger.critical(f'Not supported quantization method for inferable quantizers.')  # pragma: no cover
    else:
        Logger.critical(f'{quantization_target} is not supported')  # pragma: no cover


def get_weights_quantizer_for_node(node: BaseNode) -> BaseKerasInferableQuantizer:
    """
    Get weights quantizer for a node.
    Args:
        node: Node to create a weight quantizer for.
    Returns:
        Quantizer for the node's weights.
    """
    if node.final_weights_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final weights quantization configuration')  # pragma:
        # no cover
    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method

    quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Weights,
                                                      weights_quantization_method,
                                                      BaseKerasInferableQuantizer)
    kwargs = get_inferable_quantizer_kwargs(node_w_qc, QuantizationTarget.Weights)

    return quantier_for_node(**kwargs)


def get_activations_quantizer_for_node(node: BaseNode) -> BaseKerasInferableQuantizer:
    """
    Get activation quantizer for a node.
    Args:
        node: Node to create an activation quantizer for.
    Returns:
        Quantizer for the node's activations.
    """
    if node.final_activation_quantization_cfg is None:
        Logger.critical(f'Can not set quantizer for a node with no final activation quantization configuration')  #
        # pragma: no cover
    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method

    quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Activation,
                                                      activation_quantization_method,
                                                      BaseKerasInferableQuantizer)
    kwargs = get_inferable_quantizer_kwargs(node_act_qc, QuantizationTarget.Activation)

    return quantier_for_node(**kwargs)