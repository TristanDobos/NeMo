# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary spherical quantization (see https://arxiv.org/abs/2406.07548)."""

# Adapted from:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/3e4ce165774d3f5944f12ffb5ccd02848bb38df6/vector_quantize_pytorch/lookup_free_quantization.py

import math
from typing import Optional, Tuple

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.modules.audio_codec_modules import return_first_samples
import torch
from torch import Tensor, nn
from nemo.collections.common.parts.utils import   mask_sequence_tensor
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    Index,
    LengthsType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from einops import rearrange

__all__ = ["BinarySphericalQuantizer"]


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 4096) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = int(math.log2(codebook_size))

        if 2 ** self.dim != codebook_size:
            raise ValueError(
                f"codebook_size must be a power of 2, got {codebook_size}"
            )

        self.register_buffer(
            "codebook_value",
            torch.tensor(1 / math.sqrt(self.dim)),
            persistent=False,
        )
        self.register_buffer(
            "mask",
            2 ** torch.arange(self.dim - 1, -1, -1),
            persistent=False,
        )

        all_codes = torch.arange(codebook_size)
        bits = (all_codes[..., None].int() & self.mask) != 0
        codebook = self._bits_to_codes(bits.to(torch.float32)) * self.codebook_value
        self.register_buffer("codebook", codebook, persistent=False)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(("B", "D", "T"), EncodedRepresentation()),
            "input_len": NeuralType(tuple("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "dequantized": NeuralType(("B", "D", "T"), EncodedRepresentation()),
            "indices": NeuralType(("D", "B", "T"), Index()),
        }

    @typecheck()
    def forward(
        self, inputs: torch.Tensor, input_len: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: [B, D, T]
        returns:
          dequantized: [B, D, T]
          indices:     [D, B, T] (where D=1)
        """
        # [B, D, T] -> [B, T, D]
        lats_bt_d = rearrange(inputs, "b d t -> b t d")

        # [B, T, D] -> [B, T]
        toks_bt = self.lats_to_toks(lats_bt_d)

        # [B, T] -> [B, T, D]
        dequantized_bt_d = self.toks_to_codes(toks_bt)

        # [B, T, D] -> [B, D, T]
        dequantized = rearrange(dequantized_bt_d, "b t d -> b d t")

        # [B, T] -> [1, B, T] 
        # Crucial: Ensure this is (D, B, T) for NeMo's internal validation
        indices = toks_bt.unsqueeze(0)

        if input_len is not None:
            dequantized = mask_sequence_tensor(dequantized, input_len)

        return dequantized, indices

    @typecheck(
        input_types={
            "inputs": NeuralType(("B", "D", "T"), EncodedRepresentation()),
            "input_len": NeuralType(tuple("B"), LengthsType(), optional=True),
        },
        output_types={"indices": NeuralType(("D", "B", "T"), Index())},
    )
    def encode(
        self, inputs: Tensor, input_len: Optional[Tensor] = None
    ) -> Tensor:
        dequantized, indices = self(inputs=inputs, input_len=input_len)
        return indices

    @typecheck(
        input_types={
            "indices": NeuralType(("D", "B", "T"), Index()),
            "input_len": NeuralType(tuple("B"), LengthsType(), optional=True),
        },
        output_types={
            "dequantized": NeuralType(("B", "D", "T"), EncodedRepresentation()),
        },
    )
    def decode(
        self, indices: Tensor, input_len: Optional[Tensor] = None
    ) -> Tensor:
        if indices.size(0) != 1:
            raise ValueError(
                f"Expected a single codebook, got {indices.size(0)} for shape {indices.shape}."
            )

        indices_bt = indices.squeeze(0)                     # [B, T]
        dequantized_bt_d = self.toks_to_codes(indices_bt)  # [B, T, D]
        dequantized = rearrange(dequantized_bt_d, "B T D -> B D T")

        if input_len is not None:
            dequantized = mask_sequence_tensor(dequantized, input_len)

        return dequantized

    @torch.jit.export
    def lats_to_codes(self, lats: Tensor) -> Tensor:
        return torch.where(lats > 0, self.codebook_value, -self.codebook_value)

    @torch.jit.export
    def lats_to_toks(self, lats: Tensor) -> Tensor:
        return self.codes_to_toks(lats)

    @torch.jit.export
    def codes_to_toks(self, codes: Tensor) -> Tensor:
        return ((codes > 0) * self.mask).sum(dim=-1)

    @torch.jit.export
    def toks_to_codes(self, toks: Tensor) -> Tensor:
        bits = ((toks[..., None] // self.mask) % 2).to(self.codebook.dtype)
        return self._bits_to_codes(bits) * self.codebook_value

    def _bits_to_codes(self, bits: Tensor) -> Tensor:
        return bits * 2 - 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(codebook_size={self.codebook_size})"


def test_model() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = BinarySphericalQuantizer().to(device)
    print(model)
    print(
        f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M"
    )

    lats = torch.randn(B, T, model.dim, device=device)
    toks, codes = model(lats)
    codes2 = model.lats_to_codes(lats)
    toks2 = model.lats_to_toks(lats)
    toks3 = model.codes_to_toks(codes)
    assert (toks == toks2).all()
    assert (toks == toks3).all()
    assert (codes == codes2).all()
    model_jit = torch.jit.script(model)
    toks_jit, codes_jit = model_jit(lats)
    assert (toks == toks_jit).all()
    assert (codes == codes_jit).all()

    print("Model test passed")


def test_batch_invariance() -> "None":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 10
    T = 50
    model = BinarySphericalQuantizer().to(device)

    lats = torch.randn(B, T, model.dim, device=device)
    batch_toks, batch_codes = model(lats)

    all_single_toks, all_single_codes = [], []
    for i in range(B):
        single_toks, single_codes = model(lats[i][None])
        all_single_toks.append(single_toks)
        all_single_codes.append(single_codes)
    all_single_toks = torch.cat(all_single_toks)
    all_single_codes = torch.cat(all_single_codes)

    assert (batch_toks == all_single_toks).all()
    assert (batch_codes == all_single_codes).all()

    print("Batch invariance test passed")


@torch.no_grad()
def test_onnx() -> "None":
    import io
    import warnings

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 3
    T = 50
    model = BinarySphericalQuantizer().eval().to(device)

    lats = torch.randn(B, T, model.dim, device=device)

    f = io.BytesIO()
    torch.onnx.export(
        model,
        (lats,),
        f,
        input_names=["lats"],
        output_names=["toks", "codes"],
        dynamic_axes={
            "lats": {0: "batch", 1: "latent_time"},
            "toks": {0: "batch", 1: "latent_time"},
            "codes": {0: "batch", 1: "latent_time"},
        },
    )
    onnx_bytes = f.getvalue()

    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("`pip install onnxruntime` to test ONNX")
        return

    lats = torch.randn(2 * B, 2 * T, model.dim, device=device)

    session = ort.InferenceSession(onnx_bytes)
    inputs_ort = dict(zip([x.name for x in session.get_inputs()], [lats.cpu().numpy()]))
    outputs_ort = session.run([x.name for x in session.get_outputs()], inputs_ort)
    toks, codes = model(lats)

    assert (toks.cpu() == torch.tensor(outputs_ort[0])).all()
    assert (codes.cpu() == torch.tensor(outputs_ort[1])).all()

    print("ONNX test passed")


if __name__ == "__main__":
    test_model()
    test_batch_invariance()
    test_onnx()
