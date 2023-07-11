# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Callable, Tuple
from packaging import version

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import TENSORFLOW, FOUND_TF
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities

LR_DEFAULT = 0.15
LR_REST_DEFAULT = 1e-4
LR_BIAS_DEFAULT = 1e-4
LR_QUANTIZATION_PARAM_DEFAULT = 1e-3
GPTQ_MOMENTUM = 0.9

if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.gptq.keras.gptq_keras_implementation import GPTQKerasImplemantation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.gptq.keras.gptq_loss import GPTQMultipleTensorsLoss
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.exporter.model_wrapper import get_exportable_keras_model
    from model_compression_toolkit import get_target_platform_capabilities

    # As from TF2.9 optimizers package is changed
    if version.parse(tf.__version__) < version.parse("2.9"):
        from keras.optimizer_v2.optimizer_v2 import OptimizerV2
    else:
        from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def get_keras_gptq_config(n_epochs: int,
                              optimizer: OptimizerV2 = tf.keras.optimizers.Adam(learning_rate=LR_DEFAULT),
                              optimizer_rest: OptimizerV2 = tf.keras.optimizers.Adam(learning_rate=LR_REST_DEFAULT),
                              loss: Callable = GPTQMultipleTensorsLoss(),
                              log_function: Callable = None,
                              use_hessian_based_weights: bool = True) -> GradientPTQConfigV2:
        """
        Create a GradientPTQConfigV2 instance for Keras models.

        args:
            n_epochs (int): Number of epochs for running the representative dataset for fine-tuning.
            optimizer (OptimizerV2): Keras optimizer to use for fine-tuning for auxiliry variable with a default learning rate set to 0.2.
            optimizer_rest (OptimizerV2): Keras optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. should accept 4 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors, the 3rd is a list of quantized weights and the 4th is a list of float weights.
            log_function (Callable): Function to log information about the gptq process.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.

        returns:
            a GradientPTQConfigV2 object to use when fine-tuning the quantized model using gptq.

        Examples:

            Import MCT and TensorFlow:

            >>> import model_compression_toolkit as mct
            >>> import tensorflow as tf

            Create a GradientPTQConfigV2 to run for 5 epochs:

            >>> gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=5)

            Other Tensorflow optimizers can be passed:

            >>> gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=3, optimizer=tf.keras.optimizers.Nadam())

            The configuration can be passed to :func:`~model_compression_toolkit.keras_post_training_quantization` in order to quantize a keras model using gptq.

        """
        bias_optimizer = tf.keras.optimizers.SGD(learning_rate=LR_BIAS_DEFAULT,
                                                 momentum=GPTQ_MOMENTUM)
        return GradientPTQConfigV2(n_epochs,
                                   optimizer,
                                   optimizer_rest=optimizer_rest,
                                   loss=loss,
                                   log_function=log_function,
                                   train_bias=True,
                                   optimizer_bias=bias_optimizer,
                                   use_hessian_based_weights=use_hessian_based_weights)


    def keras_gradient_post_training_quantization_experimental(in_model: Model,
                                                               representative_data_gen: Callable,
                                                               gptq_config: GradientPTQConfigV2,
                                                               gptq_representative_data_gen: Callable = None,
                                                               target_kpi: KPI = None,
                                                               core_config: CoreConfig = CoreConfig(),
                                                               fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                                               target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC,
                                                               new_experimental_exporter: bool = True) -> Tuple[Model, UserInformation]:
        """
        Quantize a trained Keras model using post-training quantization. The model is quantized using a
        symmetric constraint quantization thresholds (power of two).
        The model is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        For each possible bit width (per layer) a threshold is then being calculated using the collected
        statistics. Then, if given a mixed precision config in the core_config, using an ILP solver we find
        a mixed-precision configuration, and set a bit-width for each layer. The model is then quantized
        (both coefficients and activations by default).
        In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
        is set (in bytes).
        Then, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized models, and minimizing the observed
        loss.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            gptq_config (GradientPTQConfigV2): Configuration for using gptq (e.g. optimizer).
            gptq_representative_data_gen (Callable): Dataset used for GPTQ training. If None defaults to representative_data_gen
            target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to.
            new_experimental_exporter (bool): Whether to wrap the quantized model using quantization information or not. Enabled by default. Experimental and subject to future changes.

        Returns:

            A quantized model and information the user may need to handle the quantized model.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 224, 224, 3))]

            Create an MCT core config, containing the quantization configuration:

            >>> config = mct.core.CoreConfig()

            If mixed precision is desired, create an MCT core config with a mixed-precision configuration, to quantize a model
            with different bitwidths for different layers.
            The candidates bitwidth for quantization should be defined in the target platform model:

            >>> config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1))

            For mixed-precision set a target KPI object:
            Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
            that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
            while the bias will not):

            >>> kpi = mct.core.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

            Create GPTQ config:

            >>> gptq_config = mct.gptq.get_keras_gptq_config(n_epochs=1)

            Pass the model with the representative dataset generator to get a quantized model:

            >>> quantized_model, quantization_info = mct.gptq.keras_gradient_post_training_quantization_experimental(model, repr_datagen, gptq_config, target_kpi=kpi, core_config=config)

        """
        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use keras_post_training_quantization "
                                    "API, or pass a valid mixed precision configuration.")  # pragma: no cover

            Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = GPTQKerasImplemantation()

        tg, bit_widths_config = core_runner(in_model=in_model,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            target_kpi=target_kpi,
                                            tb_w=tb_w)

        tg_gptq = gptq_runner(tg,
                              core_config,
                              gptq_config,
                              representative_data_gen,
                              gptq_representative_data_gen if gptq_representative_data_gen else representative_data_gen,
                              fw_info,
                              fw_impl,
                              tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg_gptq, fw_impl, fw_info)

        if new_experimental_exporter:
            Logger.warning('Using new experimental wrapped and ready for export models. To '
                           'disable it, please set new_experimental_exporter to False when '
                           'calling keras_gradient_post_training_quantization_experimental. '
                           'If you encounter an issue please file a bug.')

            return get_exportable_keras_model(tg_gptq)

        return export_model(tg_gptq,
                            fw_info,
                            fw_impl,
                            tb_w,
                            bit_widths_config)

else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def get_keras_gptq_config(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_post_training_quantization_mixed_precision. '
                        'Could not find Tensorflow package.')  # pragma: no cover


    def keras_gradient_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_gradient_post_training_quantization_experimental. '
                        'Could not find Tensorflow package.')  # pragma: no cover
