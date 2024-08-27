import gym
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger

from ivn_proc.environment import IThorEnvironment
from ivn_proc.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
    DIRECTIONAL_AHEAD_PUSH,
    DIRECTIONAL_BACK_PUSH,
    DIRECTIONAL_RIGHT_PUSH,
    DIRECTIONAL_LEFT_PUSH,
    PICK_UP,
    DROP
)

from allenact_plugins.ithor_plugin.ithor_util import (
    round_to_factor,
    include_object_data,
)

class ObstaclesNavTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH, PICK_UP, DROP,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_target(self.task_info["target"])
        self.last_geodesic_distance_ = self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])
        self.last_tget_in_path = False

        self.optimal_distance = self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])
        self.optimal_distance_ = self.env.distance_to_target(self.task_info["target"])
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

        self.push_success = 0
        self.pick_success = 0
        self.distance_change: float = 0.0
        self.push_moves = 0
        self.pick_moves = 0
        self.effective_moves = 0
        self.push_val = False
        self.pick_val = False
        self.action_str = None
        self.action = None
        self.last_action_success = False
        self.inter_reward = 0
        self.inter_reward_ = 0
        self.nav_reward = 0
        self.inter_moves=0
        self.decision_reward = 0
        self.episode_length = 0

        self.greedy_expert = None

        self.mode = 'nav'
        self.start = True
        self.last_mode = 'nav'
        self.target_node = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action) -> RLStepResult:
        self.episode_length += 1

        action = cast(int, action)

        action_str = self.action_names()[action]

        if self.mode == 'pick' or self.mode == 'move':
            if action_str != END:
                self.inter_moves += 1
        else:
            self.inter_moves =0

        if self.inter_moves > 10:
            action_str = END
            self.inter_moves=0

        self.action_str = action_str
        self.action = action

        if action_str == END:
            if self.mode == 'nav' and not self.start:
                self._took_end_action = True
                self._success = self._is_goal_in_range()
                self.last_action_success = self._success
            else:
                self.mode = 'nav'
                self.env.step({"action": "Done"})
        elif action_str == PICK_UP:
            if self.mode == 'nav':
                self.mode = 'pick'
            self.pick_moves += 1
            obj = self.env.pickupable_closest_obj_by_types(self.task_info["obstacles_types"])
            self.pick_val = obj is not None
            if obj != None:
                self.env.step({"action": action_str,
                               "objectId": obj["objectId"],
                               })
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False
            if self.last_action_success:
                self.pick_success += 1

        elif action_str == DROP:
            self.pick_moves += 1
            self.env.step({"action": action_str,
                           "forceAction": True})
            self.last_action_success = self.env.last_action_success
            if self.last_action_success:
                self.pick_success += 1
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            if self.mode == 'nav':
                self.mode = 'move'
            angle = [0.001, 180, 90, 270][action - 5]
            obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
            self.push_val = obj is not None
            if obj != None:
                self.env.step({"action": action_str,
                               "objectId": obj["objectId"],
                               "moveMagnitude": obj["mass"] * 100,
                               "pushAngle": angle,
                               "autoSimulation": False})
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False

            self.push_moves += 1
            if self.last_action_success:
                self.push_success += 1

        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        self.start = False

        return step_result

    def extra_terminal(self) -> bool:
        # nav
        goal_in_range = self._is_goal_in_range()
        if isinstance(goal_in_range, bool):
            if goal_in_range:
                self._success = True
            return goal_in_range
        else:
            return False

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        # dist = self.dist_to_target()
        dist = self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance

        if self.action_str != MOVE_AHEAD:
            if self.last_geodesic_distance == -1.0 and geodesic_distance > 0:
                rew += 1.0

        self.last_geodesic_distance = geodesic_distance

        if self.action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH, PICK_UP, DROP]:
            self.inter_reward += rew

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        geodesic_distance_ = self.dist_to_target_()
        geodesic_distance_1 = self.env.distance_to_target(self.task_info['all_node_paths'][0][self.target_node])
        if geodesic_distance_ == -1.0:
            geodesic_distance_ = self.last_geodesic_distance_
        if self.action_str in [MOVE_AHEAD]:
            if 0 <= geodesic_distance_1 <= 0.5 and self.target_node < len(self.task_info['all_node_paths'][0])-1:
                self.target_node += 1
            if (
                    self.last_geodesic_distance_ > -0.5 and geodesic_distance_ > -0.5
            ):  # (robothor limits)
                reward += self.last_geodesic_distance_ - geodesic_distance_
                self.nav_reward += reward
        self.last_geodesic_distance_ = geodesic_distance_
        return reward


    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]
        reward += self.shaping_by_path()
        reward += self.shaping()
        tmp = reward
        self.decision_reward += reward - tmp

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else 0
                )

        self._rewards.append(float(reward))
        self.last_mode = self.mode
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def sts(self):
        if not self._success:
            return 0.0
        res = (self.optimal_distance/self.env._grid_size)/self.episode_length
        return res

    def dist_to_target(self):
        return self.env.distance_to_target(self.task_info["target"])

    def dist_to_target_(self):
        return self.env.distance_to_point(self.task_info['scene'], self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target_()
        if dist2tget == -1:
            dist2tget = self.last_geodesic_distance

        spl = self.spl()
        sts = self.sts()

        return {
            **super(ObstaclesNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            'sts': sts,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError
