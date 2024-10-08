import glob, os, gym, torch
import numpy as np
from abc import ABC
from math import ceil
from typing import Dict, Any, List, Optional, Sequence
import prior
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler

from interactive_navigation_constants import ABS_PATH_OF_INTERACTIVE_NAVIGATION_TOP_LEVEL_DIR
from ivn_proc.task_samplers import ObstaclesNavDatasetTaskSampler
from ivn_proc.tasks_hier import ObstaclesNavTask
from configs.base import BaseConfig


class ObstaclesNavBaseConfig(BaseConfig, ABC):
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

    def __init__(self):
        super().__init__()
        self.ENV_ARGS = dict(
            player_screen_width=self.SCREEN_SIZE,
            player_screen_height=self.SCREEN_SIZE,
            using_mask_rcnn=False,
            mask_rcnn_dir="pretrained_model_ckpts/maskRcnn/model.pth",
            thor_commit_id=self.COMMIT_ID,
        )
        self.datasets_10k=prior.load_dataset("procthor-10k")
        self.NUM_PROCESSES = 12
        self.TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
        self.VALID_GPU_IDS = [torch.cuda.device_count() - 1]
        self.TEST_GPU_IDS = [torch.cuda.device_count() - 1]

        self.TRAIN_DATASET_DIR = os.path.join(
            ABS_PATH_OF_INTERACTIVE_NAVIGATION_TOP_LEVEL_DIR, "datasets/ProcIVN/train"
        )
        self.VAL_DATASET_DIR = os.path.join(
            ABS_PATH_OF_INTERACTIVE_NAVIGATION_TOP_LEVEL_DIR, "datasets/ProcIVN/val"
        )

        self.TARGET_TYPES = None
        self.SENSORS = None
        self.OBSTACLES_TYPES = ["ArmChair", "DogBed", "Box", "Chair", "Desk", "DiningTable", "SideTable", "Sofa",
                                "Stool", "Television", "Pillow", "Bread", "Apple", "AlarmClock", "Lettuce",
                                "GarbageCan", "Laptop", "Microwave", "Pot", "Tomato"]

    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            self.NUM_PROCESSES, ndevices
        )
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
            sampler_devices = self.TRAIN_GPU_IDS * workers_per_device
        elif mode == "valid":
            nprocesses = 0
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            nprocesses = 10
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(self.SENSORS).observation_spaces,
                preprocessors=self.PREPROCESSORS,
            )
            if mode == "train"
               or (
                       (isinstance(nprocesses, int) and nprocesses > 0)
                       or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
               )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=gpu_ids,
            sampler_devices=sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObstaclesNavDatasetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        path = os.path.join(scenes_dir, "*.json.gz")
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObstaclesNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        res["env_args"] = {}
        res['env_args']['prior_dataset'] = self.datasets_10k['train']
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("2.0")
            if devices is not None and len(devices) > 0
            else None
        )
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res['env_args']['prior_dataset'] = self.datasets_10k['val']
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("2.0")
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res['env_args']['prior_dataset'] = self.datasets_10k['val']
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("2.0")
            if devices is not None and len(devices) > 0
            else None
        )
        return res
