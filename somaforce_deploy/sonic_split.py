"""Official GEAR-SONIC encoder/decoder inference boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .sonic import (
    SONIC_ACTION_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_PROPRIOCEPTION_DIM,
    SONIC_TOKEN_DIM,
    validate_sonic_bundle,
)


class SonicSplitRuntime:
    """Run the official encoder and decoder ONNX graphs in lockstep."""

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path) -> None:
        import onnxruntime as ort

        self.encoder_path = Path(encoder_path).expanduser().resolve()
        self.decoder_path = Path(decoder_path).expanduser().resolve()
        validate_sonic_bundle(self.encoder_path, self.decoder_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(__import__("os").environ.get("SIM2REAL_ORT_NUM_THREADS", "1"))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.encoder = ort.InferenceSession(
            str(self.encoder_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(self.decoder_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        encoder_input = np.asarray(inputs["g1_input"], dtype=np.float32)
        proprioception = np.asarray(inputs["proprioception"], dtype=np.float32)
        if encoder_input.shape != (1, SONIC_ENCODER_INPUT_DIM):
            raise ValueError(f"Sonic encoder input must be {(1, SONIC_ENCODER_INPUT_DIM)}, got {encoder_input.shape}")
        if proprioception.shape != (1, SONIC_PROPRIOCEPTION_DIM):
            raise ValueError(
                f"Sonic proprioception must be {(1, SONIC_PROPRIOCEPTION_DIM)}, got {proprioception.shape}"
            )
        token = np.asarray(
            self.encoder.run(None, {self.encoder.get_inputs()[0].name: encoder_input})[0],
            dtype=np.float32,
        )
        if token.shape != (1, SONIC_TOKEN_DIM):
            raise RuntimeError(f"Sonic encoder returned {token.shape}, expected {(1, SONIC_TOKEN_DIM)}")
        decoder_input = np.concatenate((token, proprioception), axis=-1)
        action = np.asarray(
            self.decoder.run(None, {self.decoder.get_inputs()[0].name: decoder_input})[0],
            dtype=np.float32,
        )
        if action.shape != (1, SONIC_ACTION_DIM):
            raise RuntimeError(f"Sonic decoder returned {action.shape}, expected {(1, SONIC_ACTION_DIM)}")
        return {"action": action, "token": token}
