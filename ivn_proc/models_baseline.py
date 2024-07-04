import typing, gym, torch, os, time, math
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union, List, cast, Sequence
from gym.spaces.dict import Dict as SpaceDict
from torchvision import utils as vutils
import pickle
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from torch.nn.parameter import Parameter
import pickle
from torchvision import models
import torchvision
# from roi_align import CropAndResize
import torch.nn.functional as F

from ivn_proc.intervention_classifier import Interventional_Classifier

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
action_space_num = {'decision': 4, 'nav': 5, 'pick': 5, 'move': 5}


class LinearActorHead_(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x) # type:ignore

        # noinspection PyArgumentList
        return x  # logits are [step, sampler, ...]


class ObstaclesNavRGBDActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        num_rnn_layers=1,
        rnn_type="GRU",
        mode='nav',
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.intent_size = 12
        self.intent_embedding_size = 12
        self.embed_intent = True

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True
            # for p in self.sensor_fuser.parameters():
            #     p.requires_grad = False

            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")
        # for p in self.visual_encoder.parameters():
        #     p.requires_grad = False

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.state_encoder_ = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        # for p in self.state_encoder.parameters():
        #     p.requires_grad = False

        self.state_encoder_intent = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + self.intent_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        for p in self.state_encoder_intent.parameters():
            p.requires_grad = False

        self.action_space = action_space
        self.mode = mode
        # global policy
        self.actor = LinearActorHead(self._hidden_size, 10)
        self.critic = LinearCriticHead(self._hidden_size)

        self.actor_ = LinearActorHead(self._hidden_size, 12)
        self.critic_ = LinearCriticHead(self._hidden_size)

        self.actor_intent = LinearActorHead(self._hidden_size, 12)
        for p in self.actor_intent.parameters():
            p.requires_grad = False

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )
        if self.embed_intent:
            self.intent_embedding = nn.Linear(
                self.intent_size, self.intent_embedding_size
            )
        if self.embed_intent:
            self.intent_embedding_ = nn.Linear(
                self.intent_size, self.intent_embedding_size
            )
        for p in self.intent_embedding_.parameters():
            p.requires_grad = False

        self.data_num=0
        self.train()
        self.mode = [0]*40
        self.last_step_size = 1
        self.count = 0

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            rnn_=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            rnn_intent=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,  #步数step
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        # synchronize the intent policy
        if self.last_step_size == 30:
            self.count += 1
        # # if self.count%12 == 0 and self.count != 0 and self.last_step_size == 30:
        # if self.count % 15 == 0 and self.count != 0 and self.last_step_size == 30:
            self.state_encoder_intent.load_state_dict(self.state_encoder_.state_dict())
            self.actor_intent.load_state_dict(self.actor_.state_dict())
            self.intent_embedding_.load_state_dict(self.intent_embedding.state_dict())

        observations['rgb'] = torch.where(torch.isinf(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                          observations['rgb'])
        observations['depth'] = torch.where(torch.isinf(observations['depth']),
                                            torch.full_like(observations['depth'], 1),
                                            observations['depth'])
        observations['rgb'] = torch.where(torch.isnan(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                          observations['rgb'])
        observations['depth'] = torch.where(torch.isnan(observations['depth']),
                                            torch.full_like(observations['depth'], 1),
                                            observations['depth'])

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)

            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)

        device_ = x.device
        void_intent = torch.full((x.shape[0],x.shape[1],self.intent_size), 1/12).to(device_)

        if self.embed_intent:
            void_intent = self.intent_embedding_(void_intent)

        x_intent = torch.cat((void_intent,x), dim=2)

        x_intent, rnn_hidden_states_intent = self.state_encoder_intent(x_intent, memory.tensor("rnn_intent"), masks)
        intent = self.actor_intent(x_intent)
        intent = torch.softmax(intent.logits, dim=-1)
        if self.embed_intent:
            intent = self.intent_embedding(intent)

        x = torch.cat((intent, x), dim=2)
        x, rnn_hidden_states = self.state_encoder_(x, memory.tensor("rnn_"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor_(x), values=self.critic_(x), extras={}
        )

        memory.set_tensor("rnn_", rnn_hidden_states)
        memory.set_tensor("rnn_intent", rnn_hidden_states_intent)

        self.last_step_size = x.shape[0]

        return ac_output, memory