import mujoco
import matplotlib.pyplot as plt

XML = "walker2d_sim2.xml"
JOINT = "thigh_joint"

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, JOINT
)

actuator_id = next(
    i for i in range(model.nu)
    if model.actuator_trnid[i, 0] == joint_id
)

act_id = model.actuator_actadr[actuator_id]

times, commands, actual = [], [], []

while data.time < 12:
    u = 1.0 if 1 <= data.time < 7 else 0.0

    data.ctrl[:] = 0
    data.ctrl[actuator_id] = u
    mujoco.mj_step(model, data)

    # filterexact: read filtered activation.
    # normal motor: actual command equals ctrl.
    ua = data.act[act_id] if act_id >= 0 else data.ctrl[actuator_id]

    times.append(data.time)
    commands.append(u)
    actual.append(ua)

plt.step(times, commands, where="post",
         linestyle="--", label=r"Command $u(t)$")
plt.plot(times, actual, label=r"Actual $u_a(t)$")
plt.xlabel("Time [s]")
plt.ylabel("Actuator command")
plt.legend()
plt.tight_layout()
plt.savefig(f"{JOINT}_{XML}.png", dpi=300, bbox_inches="tight")
plt.close()