"""SomaForce deployment contracts, runtime, and policy composition."""
from .artifacts import ArtifactEntry, ArtifactManifest, sha256_file
from .contracts import *
from .nominal import HDMIStudentNominal, HDMIStudentTwoStageNominal, NominalPolicy, SonicNominal
from .residual import CrossResidual, compose_action
from .sonic import (
    SONIC_ACTION_DIM,
    SONIC_CONTROL_HZ,
    SONIC_FUTURE_STEPS,
    SONIC_G1_INPUT_DIM,
    SONIC_ENCODER_INPUT_DIM,
    SONIC_ACTION_JOINT_NAMES,
    SONIC_PROPRIOCEPTION_DIM,
    SONIC_TOKEN_DIM,
    select_cross_action,
    validate_sonic_onnx,
    validate_sonic_bundle,
)
from .sonic_split import SonicSplitRuntime
from .runtime import ActionHistory, DeploymentStack, StepResult
