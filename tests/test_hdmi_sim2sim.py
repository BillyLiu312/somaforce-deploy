from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco

from somaforce_deploy.hdmi_sim2sim import TASKS, materialize_scene


def test_hdmi_task_profiles_have_expected_cycles():
    assert TASKS["push_door_hand"].policy_steps == 573
    assert TASKS["push_box"].policy_steps == 792
    assert TASKS["move_largebox"].policy_steps == 199


def test_hdmi_task_scene_materialization_contracts():
    upstream = Path("/tmp/egalahad-sim2real-hdmi-1855")
    hdmi = Path(__file__).resolve().parents[2] / "HDMI"
    if not upstream.is_dir() or not hdmi.is_dir():
        return
    for task in TASKS.values():
        with TemporaryDirectory() as temp_dir:
            scene = materialize_scene(
                task,
                upstream_root=upstream,
                hdmi_root=hdmi,
                output_dir=Path(temp_dir),
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
            joint_names = {model.joint(i).name for i in range(model.njnt)}
            sensor_names = {model.sensor(i).name for i in range(model.nsensor)}
            assert task.primary_object_joint in joint_names
            for body_name in task.publish_object_names:
                assert f"{body_name}_pos" in sensor_names
                assert f"{body_name}_quat" in sensor_names
