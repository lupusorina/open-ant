import mujoco

# Load and compile the Hopper XML model file
model = mujoco.MjModel.from_xml_path("walker2d_sim2.xml")

# Sum the mass of all bodies in the compiled model
total_mass = model.body_mass.sum()

print(f"Total Mass: {total_mass} kg")

# Print individual body masses
for i in range(model.nbody):
  body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
  print(f"Body '{body_name}': {model.body_mass[i]} kg")