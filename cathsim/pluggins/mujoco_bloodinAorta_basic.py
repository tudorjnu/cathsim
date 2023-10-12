import mujoco
from mujoco import viewer


model = mujoco.MjModel.from_xml_path('./pluggins/mujoco_xml_model_blood/blood_flow_in_aorta.xml')
viewer.launch(model=model)


