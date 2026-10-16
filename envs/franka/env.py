"""Menagerie Panda with a physically grasped block, open bin and dynamic glass mug."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .planner import BLOCK_HALF, BIN_HALF, DT, T, Planner
from .setup_assets import DEFAULT_MODEL

CONTROL_DT = .02
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -.7853])


def build_model(path=DEFAULT_MODEL):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m envs.franka.setup_assets")
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent / "assets"))
    root.find("option").set("timestep", "0.002")
    root.remove(root.find("keyframe"))  # free objects extend qpos beyond upstream home key
    hand = root.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="tcp", pos="0 0 0.1034", size=".005", rgba="1 0 0 1")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="box", pos=".4 0 -.025",
                  size=".65 .65 .025", rgba=".55 .43 .31 1")
    ET.SubElement(world, "camera", name="overview", pos="1.35 -1.25 1.15",
                  xyaxes=".707 .707 0 -.408 .408 .816")
    block = ET.SubElement(world, "body", name="block", pos=".45 -.23 .02")
    ET.SubElement(block, "freejoint", name="block_joint")
    ET.SubElement(block, "geom", name="block_geom", type="box", size=".02 .02 .02",
                  mass=".06", friction="2 .01 .001", rgba=".9 .25 .08 1", condim="4")
    bin_body = ET.SubElement(world, "body", name="bin", pos=".46 .25 0")
    for name, pos, size in [
        ("floor", "0 0 .005", ".10 .10 .005"),
        ("left", "-.095 0 .04", ".005 .10 .04"),
        ("right", ".095 0 .04", ".005 .10 .04"),
        ("front", "0 -.095 .04", ".09 .005 .04"),
        ("back", "0 .095 .04", ".09 .005 .04"),
    ]:
        ET.SubElement(bin_body, "geom", name="bin_" + name, type="box", pos=pos,
                      size=size, rgba=".15 .35 .5 1")
    cup = ET.SubElement(world, "body", name="cup", pos=".51 .01 0")
    ET.SubElement(cup, "freejoint", name="cup_joint")
    glass = dict(rgba=".65 .85 .95 .3", friction=".7 .005 .001")
    ET.SubElement(cup, "geom", name="cup_base", type="cylinder", size=".04 .004",
                  pos="0 0 .004", mass=".08", **glass)
    # Convex pieces form a hollow, open mug, rather than a solid collision cylinder.
    for i, a in enumerate(np.linspace(0, 2*np.pi, 20, endpoint=False)):
        ET.SubElement(cup, "geom", name=f"cup_wall_{i}", type="box",
                      pos=f"{.037*np.cos(a)} {.037*np.sin(a)} .064",
                      size=".003 .006 .056", euler=f"0 0 {a}", mass=".008", **glass)
    # Handle on the side away from the planned passage.
    for i, (p, q) in enumerate([((.038, 0, .095), (.075, 0, .095)),
                                ((.075, 0, .095), (.075, 0, .035)),
                                ((.075, 0, .035), (.038, 0, .035))]):
        ET.SubElement(cup, "geom", name=f"cup_handle_{i}", type="capsule", size=".006",
                      fromto=" ".join(map(str, (*p, *q))), mass=".01", **glass)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class FrankaEnv:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = m = build_model(model_path)
        self.data = mujoco.MjData(m)
        self.ik_data = mujoco.MjData(m)
        self.tcp = m.site("tcp").id
        self.block = m.body("block").id
        self.cup = m.body("cup").id
        self.bin = m.body("bin").id
        self.fingers = {m.body(n).id for n in ("left_finger", "right_finger")}
        self.robot_bodies = {m.body(n).id for n in [f"link{i}" for i in range(1, 8)]
                             + ["hand", "left_finger", "right_finger"]}
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.robot_bodies
                            and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.cup_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.cup]
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.target_rotation = np.diag([1., -1., -1.])

    def ik(self, target, seed, iterations=12):
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[:7] = seed
        for _ in range(iterations):
            mujoco.mj_forward(m, d)
            rot = d.site_xmat[self.tcp].reshape(3, 3)
            angle = .5 * sum(np.cross(rot[:, i], self.target_rotation[:, i]) for i in range(3))
            error = np.r_[target - d.site_xpos[self.tcp], angle]
            if np.linalg.norm(error) < 1e-5:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.tcp)
            jac = np.vstack((self.jacp[:, :7], self.jacr[:, :7]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + .0001*np.eye(6), error)
            d.qpos[:7] = np.clip(d.qpos[:7] + np.clip(dq, -.12, .12),
                                m.jnt_range[:7, 0] + .001, m.jnt_range[:7, 1] - .001)
        return d.qpos[:7].copy()

    def reset(self, scenario):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        m.body_pos[self.bin] = scenario["bin"]
        d.qpos[:7], d.qpos[7:9] = HOME, .04
        for name in ("block", "cup"):
            adr = m.jnt_qposadr[m.joint(name + "_joint").id]
            d.qpos[adr:adr+3] = scenario[name]
            d.qpos[adr+3:adr+7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        # Start above the source with a consistent downward gripper orientation.
        d.qpos[:7] = self.ik(scenario["block"] + [0, 0, .20], HOME, 100)
        d.ctrl[:7], d.ctrl[7] = d.qpos[:7], 255
        for _ in range(250):
            d.qfrc_applied[:7] = d.qfrc_bias[:7]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        d.time = 0

    def contacts(self):
        """Unexpected contact and raw force sums; normal grasp/support is allowed."""
        m, d = self.model, self.data
        collision, grip = False, set()
        forces = np.zeros(4)  # robot/object, robot/world, block/world, cup/non-table
        wrench = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = (int(m.geom_bodyid[g]) for g in (c.geom1, c.geom2))
            bodies = {a, b}
            mujoco.mj_contactForce(m, d, i, wrench)
            force = np.linalg.norm(wrench[:3])
            if self.block in bodies and bodies & self.fingers:
                grip |= bodies & self.fingers
                forces[0] += force
            elif bodies & self.robot_bodies:
                forces[1] += force
                collision |= c.dist < -1e-5
            if self.cup in bodies and bodies != {0, self.cup}:
                forces[3] += force
                collision |= c.dist < -1e-5
            if self.block in bodies and not bodies & self.robot_bodies:
                forces[2] += force
                # Resting on the table or bin floor is handled by drop/placement logic.
                names = {m.geom(int(g)).name for g in (c.geom1, c.geom2)}
                collision |= any(n.startswith("bin_") and n != "bin_floor" for n in names)
        return bool(collision), len(grip) == 2, forces

    def in_bin(self):
        d = self.data
        delta = d.xpos[self.block] - self.model.body_pos[self.bin]
        # Account for the rotated cube's footprint, not just its center.
        extent = np.abs(d.xmat[self.block].reshape(3, 3)) @ np.full(3, BLOCK_HALF)
        return bool(np.all(np.abs(delta[:2]) + extent[:2] < BIN_HALF)
                    and .005 < delta[2] < .065)

    def rollout(self, scenario, style, callback=None):
        self.reset(scenario)
        m, d = self.model, self.data
        plan = Planner(scenario, style, d.site_xpos[self.tcp].copy())
        rows = []
        lifted = False
        for t in range(T):
            collided, dropped = False, False
            force_sum = np.zeros(4)
            for control in range(round(DT / CONTROL_DT)):
                now = t*DT + control*CONTROL_DT
                target, opening, phase = plan.target(now)
                d.ctrl[:7] = self.ik(target, d.ctrl[:7])
                d.ctrl[7] = opening
                for _ in range(round(CONTROL_DT / m.opt.timestep)):
                    # Feed forward gravity/Coriolis bias; contacts still carry the block.
                    d.qfrc_applied[:7] = d.qfrc_bias[:7]
                    mujoco.mj_step(m, d)
                    hit, grasped, forces = self.contacts()
                    collided |= hit
                    force_sum += forces
                    height = d.xpos[self.block, 2]
                    lifted |= grasped and height > .06
                    # Loss of support before planned release is a drop, including slips
                    # that later happen to land in the bin. Intentional release is allowed.
                    supported_placement = phase >= 7 and self.in_bin() and height < .04
                    dropped |= lifted and phase < 8 and not grasped and not supported_placement and (
                        np.linalg.norm(d.xpos[self.block] - d.site_xpos[self.tcp]) > .065
                        or height < .035)
                    dropped |= height < -.02
            mujoco.mj_forward(m, d)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.tcp, velocity, 0)
            block_vel, cup_vel = np.zeros(6), np.zeros(6)
            for body, v in ((self.block, block_vel), (self.cup, cup_vel)):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
            gap = min(mujoco.mj_geomDistance(m, d, a, b, 2., None)
                      for a in self.robot_geoms + [m.geom("block_geom").id] for b in self.cup_geoms)
            rows.append(dict(qpos=d.qpos[:9].copy(), qvel=d.qvel[:9].copy(), ctrl=d.ctrl.copy(),
                             tcp=d.site_xpos[self.tcp].copy(), tcp_vel=velocity[3:].copy(),
                             block_pos=d.xpos[self.block].copy(), block_quat=d.xquat[self.block].copy(),
                             block_vel=block_vel, cup_pos=d.xpos[self.cup].copy(),
                             cup_quat=d.xquat[self.cup].copy(), cup_vel=cup_vel,
                             bin_pos=m.body_pos[self.bin].copy(), contact_force=force_sum / (DT/m.opt.timestep),
                             cup_distance=gap, phase=phase, collision=collided, drop=dropped))
            if callback is not None:
                callback(self)
        tr = {k: np.asarray([r[k] for r in rows], dtype=np.float32) for k in rows[0]}
        settled = np.linalg.norm(tr["block_vel"][-5:, 3:], axis=1).max() < .05
        placed = lifted and self.in_bin() and settled
        tr["success"] = np.array(placed and not tr["collision"].any() and not tr["drop"].any())
        # Missing the grasp / bin / deadline is task failure, separate from a physical drop.
        tr["incomplete"] = np.zeros(T, np.float32)
        tr["incomplete"][-1] = not placed
        return tr
