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
from model_compression_toolkit.constants import FOUND_TF, FOUND_TORCH
from model_compression_toolkit.qat.common.qat_config import (
    QATConfig,
    TrainingMethod,
)  # noqa: F401

if FOUND_TF:
    from model_compression_toolkit.qat.keras.quantization_facade import (  # noqa: F401
        keras_quantization_aware_training_finalize,
        keras_quantization_aware_training_init,
    )

if FOUND_TORCH:
    from model_compression_toolkit.qat.pytorch.quantization_facade import (  # noqa: F401
        pytorch_quantization_aware_training_finalize,
        pytorch_quantization_aware_training_init,
    )
