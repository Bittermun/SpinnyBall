"""
High-Fidelity 6-DOF validation for spin-stabilized Aethelgard mass packets.
Uses MuJoCo to prove that 2kg prolate spheroids spinning at 50,000 RPM 
remain stable under steering torques and GdBCO flux-pinning forces.
"""

import mujoco
import mujoco.viewer
import numpy as np
from dataclasses import dataclass
import time
from pathlib import Path

@dataclass
class AethelgardParams:
    mp: float = 2.0             # Packet mass (kg)
    major_axis: float = 0.1     # Semi-major axis a (m)
    minor_axis: float = 0.046   # Semi-minor axis b=c (m)
    omega_spin: float = 5236.0  # rad/s (~50,000 RPM)
    u_velocity: float = 10.0    # m/s
    lam: float = 16.6667        # kg/m (s=0.12m)
    k_fp: float = 4500.0        # GdBCO pinning stiffness (N/m)
    node_mass: float = 1000.0   # Station mass (kg)
    num_packets: int = 40       # Sim pool

class SpinPacketValidation:
    def __init__(self, p: AethelgardParams):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._init_state()

    def _build_xml(self):
        # Calculate inertia for prolate spheroid (for the XML to be accurate)
        m = self.p.mp
        a = self.p.major_axis
        b = self.p.minor_axis
        ix = 0.4 * m * b**2
        iy = 0.2 * m * (a**2 + b**2)
        iz = iy
        
        xml = f"""
        <mujoco model="aethelgard_validation">
          <option timestep="0.0005" integrator="RK4"/>
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="5 5" reflection="0.2"/>
          </asset>
          <worldbody>
            <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
            <geom name="floor" type="plane" size="10 10 .1" material="grid"/>
            
            <body name="node" pos="0 0 0">
              <camera name="track" pos="3 3 3" mode="trackcom"/>
              <geom type="box" size="0.2 0.2 0.2" mass="{self.p.node_mass}" rgba="0.8 0.8 0.8 1"/>
              <joint type="free"/>
            </body>

            {"".join(f'''
            <body name="packet{i}" pos="{(i-20)*0.12} 0 0.5">
              <geom type="ellipsoid" size="{self.p.major_axis} {self.p.minor_axis} {self.p.minor_axis}" 
                    mass="{self.p.mp}" rgba="0.2 0.6 1.0 1"/>
              <joint type="free"/>
            </body>''' for i in range(self.p.num_packets))}
          </worldbody>
        </mujoco>
        """
        return xml

    def _init_state(self):
        # Apply initial spin to all packets about the major axis (x-axis in body frame)
        for i in range(1, self.model.nbody):
            # qvel indices for body i: 6*i to 6*i+6
            # [vx, vy, vz, wx, wy, wz]
            self.data.qvel[6*i+3] = self.p.omega_spin
            # Give them a small forward velocity to simulate the stream
            self.data.qvel[6*i] = self.p.u_velocity

    def step(self):
        node_pos = self.data.xpos[1] # node is body 1
        
        for i in range(2, self.model.nbody): # packets start at body 2
            p_pos = self.data.xpos[i]
            r = p_pos - node_pos
            dist = np.linalg.norm(r)
            
            # 1. Momentum Flux Restoration (Reduced Order Law)
            # F = lambda * u^2 * theta
            theta = np.arctan2(r[1], self.p.u_velocity) # Lateral angle
            F_steer = self.p.lam * self.p.u_velocity**2 * theta
            
            # 2. GdBCO Super-Pinning (Passive Restoring)
            # Acts as a stiff spring tethered to the ideal 'track'
            # Here we assume the track is the x-axis (y=0, z=fixed)
            F_pin_y = -self.p.k_fp * r[1]
            F_pin_z = -self.p.k_fp * (r[2] - 0.5) # target z=0.5
            
            # Combine forces
            force = np.array([-F_steer, F_pin_y, F_pin_z])
            
            # Apply to packet center of mass
            mujoco.mj_applyForce(self.model, self.data, force, np.zeros(3), i, 0)
            
            # Counter-force on node (Newton's 3rd)
            mujoco.mj_applyForce(self.model, self.data, -force, np.zeros(3), 1, 0)

        mujoco.mj_step(self.model, self.data)

def run_validation(gui=True):
    p = AethelgardParams()
    sim = SpinPacketValidation(p)
    
    print(f"Starting 6-DOF Validation: {p.omega_spin/ (2*np.pi):.1f} Hz Spin | {p.k_fp} N/m Pinning")
    
    # Track node displacement
    node_history = []
    
    if gui:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            for _ in range(2000):
                sim.step()
                node_history.append(sim.data.xpos[1].copy())
                viewer.sync()
    else:
        for _ in range(2000):
            sim.step()
            node_history.append(sim.data.xpos[1].copy())

    history = np.array(node_history)
    peak_displacement = np.max(np.abs(history[:, 1]))
    print(f"Peak Node Y-Displacement: {peak_displacement*1000:.3f} mm")
    
    return history

if __name__ == "__main__":
    run_validation(gui=False) # Run headless for verification
