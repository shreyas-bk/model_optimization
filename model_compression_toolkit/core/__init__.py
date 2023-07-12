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

from model_compression_toolkit.core.common.data_loader import FolderImageLoader
from model_compression_toolkit.core.common.framework_info import (
    FrameworkInfo,
    ChannelAxis,
)
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core.common import network_editors as network_editor
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.quantization import quantization_config
from model_compression_toolkit.core.common.mixed_precision import (
    mixed_precision_quantization_config,
)
from model_compression_toolkit.core.common.quantization.quantization_config import (
    QuantizationConfig,
    QuantizationErrorMethod,
    DEFAULTCONFIG,
)
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import (
    MixedPrecisionQuantizationConfig,
    MixedPrecisionQuantizationConfigV2,
)
from model_compression_toolkit.constants import FOUND_TF, FOUND_TORCH

if FOUND_TF:
    from model_compression_toolkit.core.keras.kpi_data_facade import (
        keras_kpi_data,
        keras_kpi_data_experimental,
    )

if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.kpi_data_facade import (
        pytorch_kpi_data,
        pytorch_kpi_data_experimental,
    )
