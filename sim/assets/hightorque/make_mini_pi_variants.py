"""Generate Mini Pi sim2 XMLs: heavier robot on a more slippery floor.

For each shift s, one variant scales, together:
  - every body's mass and rotational inertia by (1 + s)
  - every geom's sliding friction by (1 - s), the floor's included

Friction is scaled on *all* geoms, not just the feet: MuJoCo takes the larger
of the two geoms' friction for a contact, so lowering only the feet would
leave the floor's 1.0 in charge and change nothing. Every geom here starts at
1.0, so a contact's friction becomes exactly (1 - s). Torsional and rolling
friction are left as authored.

Inertia scales with mass so each link keeps its shape (a uniformly denser
link), not a point mass bolted on.

Reads scene.xml (which includes mini_pi.xml) via MjSpec, writes one flattened
XML per variant to generated/, plus generated/manifest.txt (one name per line).

Usage:
  python make_mini_pi_variants.py                  # s = 0.2 0.4 0.6 0.8
  python make_mini_pi_variants.py --shifts 0.3     # custom levels
"""

import argparse
import os
import re

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "scene.xml")
OUT_DIR = os.path.join(HERE, "generated")
DEFAULT_SHIFTS = (0.2, 0.4, 0.6, 0.8)


def make_spec(shift: float) -> mujoco.MjSpec:
    if not 0.0 <= shift < 1.0:
        raise ValueError(f"shift must be in [0, 1), got {shift}")
    spec = mujoco.MjSpec.from_file(TEMPLATE)

    for body in spec.bodies:
        if body.name == "world":
            continue
        body.mass *= 1.0 + shift
        body.inertia = np.asarray(body.inertia) * (1.0 + shift)

    for geom in spec.geoms:
        friction = np.array(geom.friction)
        friction[0] *= 1.0 - shift
        geom.friction = friction

    return spec


def variant_name(shift: float) -> str:
    return f"mini_pi_mass_fric_{round(shift * 100):d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shifts", type=float, nargs="+", default=list(DEFAULT_SHIFTS))
    p.add_argument("--out_dir", default=OUT_DIR)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base = mujoco.MjModel.from_xml_path(TEMPLATE)
    names = []
    for s in a.shifts:
        # to_xml() compiles, which resolves meshdir against scene.xml's folder,
        # so export with the original meshdir and repoint it in the text: the
        # output sits in a different folder from the meshes.
        xml = make_spec(s).to_xml()
        meshdir = os.path.relpath(os.path.join(HERE, "meshes"), a.out_dir)
        xml, n = re.subn(r'meshdir="[^"]*"', f'meshdir="{meshdir}"', xml, count=1)
        assert n == 1, "meshdir attribute not found in exported XML"
        name = variant_name(s)
        path = os.path.join(a.out_dir, f"{name}.xml")
        with open(path, "w") as f:
            f.write(xml)

        # Check what MuJoCo actually loads from the written file.
        m = mujoco.MjModel.from_xml_path(path)
        mass = m.body_subtreemass[0] / base.body_subtreemass[0]
        fric = np.unique(m.geom_friction[:, 0])
        print(f"wrote {name}.xml  mass x{mass:.3f}  sliding friction {fric}")
        names.append(name)

    with open(os.path.join(a.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(names) + "\n")


if __name__ == "__main__":
    main()
