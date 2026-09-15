"""Contracts shared by the official Sonic G1 export and Cross residual."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_JOINT_NAMES, G1_JOINT_NAMES, require_array


SONIC_ACTION_DIM = len(G1_JOINT_NAMES)
SONIC_ACTION_JOINT_NAMES = ACTION_JOINT_NAMES + (
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
SONIC_ENCODER_INPUT_DIM = 1762
# Backward-compatible name for the full G1 encoder vector.
SONIC_G1_INPUT_DIM = SONIC_ENCODER_INPUT_DIM
SONIC_PROPRIOCEPTION_DIM = 930
SONIC_TOKEN_DIM = 64
SONIC_FUTURE_STEPS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
SONIC_CONTROL_HZ = 50.0


def sonic_to_cross_indices(
    sonic_joint_names: tuple[str, ...] = SONIC_ACTION_JOINT_NAMES,
    cross_joint_names: tuple[str, ...] = ACTION_JOINT_NAMES,
) -> np.ndarray:
    """Return indices selecting Cross's 23 joints from Sonic's 29D action."""
    if len(set(sonic_joint_names)) != len(sonic_joint_names):
        raise ValueError("Sonic action joint names must be unique")
    missing = [name for name in cross_joint_names if name not in sonic_joint_names]
    if missing:
        raise ValueError(f"Sonic action order is missing Cross joints: {missing}")
    return np.asarray([sonic_joint_names.index(name) for name in cross_joint_names], dtype=int)


SONIC_TO_CROSS_INDICES = sonic_to_cross_indices()


def select_cross_action(action: Any) -> np.ndarray:
    """Validate a full Sonic action and return it in the Cross 23D order."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape == (SONIC_ACTION_DIM,):
        value = value[None, :]
    value = require_array(value, (1, SONIC_ACTION_DIM), "Sonic action")
    return value[:, SONIC_TO_CROSS_INDICES].copy()


def validate_sonic_onnx(path: str | Path) -> dict[str, Any]:
    """Validate the official Sonic G1 ONNX boundary without running inference."""
    import onnx

    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Sonic ONNX model is missing: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    def shape(value: Any) -> list[int | str]:
        result: list[int | str] = []
        for dim in value.type.tensor_type.shape.dim:
            result.append(int(dim.dim_value) if dim.dim_value else str(dim.dim_param))
        return result

    inputs = {value.name: shape(value) for value in model.graph.input}
    outputs = {value.name: shape(value) for value in model.graph.output}
    expected_inputs = {
        "g1_input": [SONIC_G1_INPUT_DIM],
        "proprioception": [SONIC_PROPRIOCEPTION_DIM],
    }
    expected_outputs = {"action": [SONIC_ACTION_DIM], "token": [SONIC_TOKEN_DIM]}
    if inputs != expected_inputs:
        raise ValueError(f"Sonic input contract mismatch: expected {expected_inputs}, got {inputs}")
    if outputs != expected_outputs:
        raise ValueError(
            f"Sonic output contract mismatch: expected {expected_outputs}, got {outputs}"
        )
    return {
        "path": str(model_path),
        "inputs": inputs,
        "outputs": outputs,
        "future_steps": list(SONIC_FUTURE_STEPS),
        "control_hz": SONIC_CONTROL_HZ,
        "action_joint_names": list(SONIC_ACTION_JOINT_NAMES),
        "cross_selection": SONIC_TO_CROSS_INDICES.tolist(),
    }


def validate_sonic_bundle(encoder_path: str | Path, decoder_path: str | Path) -> dict[str, Any]:
    """Validate the official split Sonic encoder/decoder pair."""
    import onnx

    def inspect(path: str | Path, expected_inputs: dict[str, list[int]], expected_outputs: dict[str, list[int]]) -> dict[str, Any]:
        model_path = Path(path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Sonic ONNX model is missing: {model_path}")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)

        def shape(value: Any) -> list[int | str]:
            return [int(dim.dim_value) if dim.dim_value else str(dim.dim_param) for dim in value.type.tensor_type.shape.dim]

        inputs = {value.name: shape(value) for value in model.graph.input}
        outputs = {value.name: shape(value) for value in model.graph.output}
        if inputs != expected_inputs or outputs != expected_outputs:
            raise ValueError(
                f"Sonic graph contract mismatch for {model_path}: "
                f"inputs={inputs}, outputs={outputs}"
            )
        return {"path": str(model_path), "sha256": __import__("hashlib").sha256(model_path.read_bytes()).hexdigest(), "inputs": inputs, "outputs": outputs}

    encoder = inspect(
        encoder_path,
        {"obs_dict": [1, SONIC_ENCODER_INPUT_DIM]},
        {"encoded_tokens": [1, SONIC_TOKEN_DIM]},
    )
    decoder = inspect(
        decoder_path,
        {"obs_dict": [1, 994]},
        {"action": [1, SONIC_ACTION_DIM]},
    )
    return {"encoder": encoder, "decoder": decoder}
