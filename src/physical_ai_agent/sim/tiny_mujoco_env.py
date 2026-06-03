from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TinyMujocoConfig:
    episode_steps: int = 64
    frame_width: int = 160
    frame_height: int = 120
    seed: int = 0


@dataclass(frozen=True)
class TinyObservation:
    step: int
    qpos: list[float]
    qvel: list[float]
    position_xy: list[float]


class TinyMujocoEnv:
    """Small deterministic MuJoCo task used to prove the local evaluation loop works."""

    def __init__(self, config: TinyMujocoConfig | None = None) -> None:
        self.config = config or TinyMujocoConfig()
        try:
            import mujoco
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MuJoCo is required for TinyMujocoEnv") from exc

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(
            """
            <mujoco model="physical_ai_agent_tiny">
              <option timestep="0.02" gravity="0 0 -9.81"/>
              <worldbody>
                <body name="puck" pos="0 0 0.03">
                  <joint name="slide_x" type="slide" axis="1 0 0" damping="0.1"/>
                  <joint name="slide_y" type="slide" axis="0 1 0" damping="0.1"/>
                  <geom name="puck_geom" type="box" size="0.035 0.035 0.025" rgba="0.1 0.45 0.9 1"/>
                </body>
              </worldbody>
              <actuator>
                <motor name="motor_x" joint="slide_x" gear="1" ctrlrange="-1 1"/>
                <motor name="motor_y" joint="slide_y" gear="1" ctrlrange="-1 1"/>
              </actuator>
            </mujoco>
            """
        )
        self.data = mujoco.MjData(self.model)
        self.step_index = 0

    @property
    def action_dim(self) -> int:
        return int(self.model.nu)

    def reset(self) -> TinyObservation:
        self._mujoco.mj_resetData(self.model, self.data)
        self.step_index = 0
        return self._observation()

    def step(self, action: list[float]) -> tuple[TinyObservation, float, bool, dict[str, Any]]:
        if len(action) != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {len(action)}")

        for index, value in enumerate(action):
            self.data.ctrl[index] = max(-1.0, min(1.0, float(value)))

        self._mujoco.mj_step(self.model, self.data)
        self.step_index += 1

        obs = self._observation()
        distance = (obs.position_xy[0] ** 2 + obs.position_xy[1] ** 2) ** 0.5
        reward = -distance
        done = self.step_index >= self.config.episode_steps
        info = {
            "completed": done,
            "finite_state": all(abs(value) < 1e6 for value in obs.qpos + obs.qvel),
            "distance_from_origin": distance,
        }
        return obs, reward, done, info

    def write_frame_ppm(self, path: str) -> None:
        obs = self._observation()
        width = self.config.frame_width
        height = self.config.frame_height
        pixels = bytearray()
        x = int(width / 2 + max(-1.0, min(1.0, obs.position_xy[0])) * width * 0.35)
        y = int(height / 2 - max(-1.0, min(1.0, obs.position_xy[1])) * height * 0.35)
        for row in range(height):
            for col in range(width):
                is_puck = abs(col - x) <= 5 and abs(row - y) <= 5
                if is_puck:
                    pixels.extend((30, 120, 230))
                elif row == height // 2 or col == width // 2:
                    pixels.extend((190, 190, 190))
                else:
                    pixels.extend((245, 245, 240))
        with open(path, "wb") as file:
            file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            file.write(pixels)

    def _observation(self) -> TinyObservation:
        qpos = [float(value) for value in self.data.qpos]
        qvel = [float(value) for value in self.data.qvel]
        return TinyObservation(
            step=self.step_index,
            qpos=qpos,
            qvel=qvel,
            position_xy=qpos[:2],
        )

