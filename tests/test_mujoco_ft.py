import mujoco
import numpy as np

from somaforce_deploy.mujoco_ft import MujocoWristFTSensor


def test_mujoco_ft_sensor_detects_object_contact():
    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body name="left_wrist_yaw_link" pos="0 0 0.1">
          <geom name="wrist" type="sphere" size="0.05"/>
        </body>
        <body name="right_wrist_yaw_link" pos="2 0 0.1">
          <geom type="sphere" size="0.05"/>
        </body>
        <body name="box" pos="0.08 0 0.1">
          <freejoint/>
          <geom name="box_geom" type="box" size="0.04 0.04 0.04" mass="1"/>
        </body>
        <geom name="floor" type="plane" size="5 5 0.1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sensor = MujocoWristFTSensor(model, contact_force_threshold=0.01)
    sample = sensor.sample(data)
    assert sample.contact_count > 0
    assert sample.contact_probability[0] > 0
    assert np.linalg.norm(sample.wrench[0, :3]) > 0
    assert sample.token.shape == (2, 14)
