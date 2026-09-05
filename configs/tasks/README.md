# Task artifacts

Keep checkpoint binaries, private F/T calibration, and sensor SDKs outside Git.
Each task artifact directory should provide `manifest.json`, the selected HDMI
student and/or Sonic ONNX, the Cross residual ONNX, frozen normalization, and
reference data. Record SHA-256 values and the nominal backend used to train the
residual. Tensor-shape equality alone does not establish Sonic compatibility.
