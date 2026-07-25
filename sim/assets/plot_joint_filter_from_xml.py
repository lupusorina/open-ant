import mujoco
import matplotlib.pyplot as plt

XML = "humanoid_sim2.xml"  # or "humanoid_sim2.xml"
JOINT = "abdomen_y"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    JOINT,
)

actuator_id = next(
    i for i in range(model.nu)
    if model.actuator_trnid[i, 0] == joint_id
)

# Read the actuator's upper control limit directly from the XML.
u_on = model.actuator_ctrlrange[actuator_id, 1]

print("Actuator:", actuator_id)
print("Control range:", model.actuator_ctrlrange[actuator_id])
print("Dynamic type:", model.actuator_dyntype[actuator_id])
print("Time constant:", model.actuator_dynprm[actuator_id, 0])

times = []
commands = []
actual_outputs = []

while data.time < 12.0:
    u = u_on if 1.0 <= data.time < 7.0 else 0.0

    data.ctrl[:] = 0.0
    data.ctrl[actuator_id] = u

    mujoco.mj_step(model, data)

    times.append(data.time)
    commands.append(u)

    # MuJoCo-computed actuator output for both motor and filterexact.
    actual_outputs.append(data.actuator_force[actuator_id])

plt.step(
    times,
    commands,
    where="post",
    linestyle="--",
    label=r"Command $u(t)$",
)

plt.plot(
    times,
    actual_outputs,
    label="Actual actuator output",
)

plt.xlabel("Time [s]")
plt.ylabel("Actuator output")
plt.legend()
plt.tight_layout()
plt.savefig(f"{JOINT}_{XML}.png", dpi=300, bbox_inches="tight")
plt.close()