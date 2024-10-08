import gym, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
torch.backends.cudnn.enabled = False


from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig, PPO
from allenact.utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay

from ivn_proc.tasks_hier import ObstaclesNavTask
from ivn_proc.models_camp import ObstaclesNavRGBDActorCriticSimpleConvRNN
from ivn_proc.sensors import (
    RGBSensorThor,
    DepthSensorIThor,
    GPSCompassSensorIThor,
)
from configs.proc10k_ObsNav.obstacles_nav_base_proc import ObstaclesNavBaseConfig


class ObstaclesNavRGBDConfig(ObstaclesNavBaseConfig):
    def __init__(self):
        super().__init__()

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb",
            ),
            DepthSensorIThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=True,
                uuid="depth",
            ),
            GPSCompassSensorIThor(),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "target_coordinates_ind",
        ]

    @classmethod
    def tag(cls):
        return "ObstaclesNav-RGBD"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(10000000)
        lr = 1e-5
        # lr = 0
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 200000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObstaclesNavRGBDActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(ObstaclesNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
            mode='nav',
        )
