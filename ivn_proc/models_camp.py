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
    #LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr,SequentialDistr,ConditionalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from torch.nn.parameter import Parameter
import pickle
from torchvision import models
import torchvision
# from roi_align import CropAndResize
import torch.nn.functional as F

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
action_space_num = {'decision': 4, 'nav': 5, 'pick': 5, 'move': 5}

class LinearActorHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore

        # noinspection PyArgumentList
        return x  # logits are [step, sampler, ...]


class LinearActorHead_(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore

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

            self.sensor_fuser_ = nn.Linear(hidden_size * 2, hidden_size)
            # for p in self.sensor_fuser_.parameters():
            #     p.requires_grad = False
            self.sensor_fuser_pick = nn.Linear(hidden_size * 2, hidden_size)
            # for p in self.sensor_fuser_pick.parameters():
            #     p.requires_grad = False
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")
        # for p in self.visual_encoder.parameters():
        #     p.requires_grad = False
        self.visual_encoder_ = SimpleCNN(observation_space, hidden_size, "rgb", "depth")
        # for p in self.visual_encoder_.parameters():
        #     p.requires_grad = False
        self.visual_encoder_pick = SimpleCNN(observation_space, hidden_size, "rgb", "depth")
        # for p in self.visual_encoder_pick.parameters():
        #     p.requires_grad = False

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        # for p in self.state_encoder.parameters():
        #     p.requires_grad = False
        self.state_encoder_ = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.state_encoder_pick = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        # for p in self.state_encoder_pick.parameters():
        #     p.requires_grad = False

        self.action_space = action_space
        self.mode = mode
        # global policy
        self.actor_master = LinearActorHead(self._hidden_size, action_space_num['decision'])
        self.critic = LinearCriticHead(self._hidden_size)
        # for p in self.critic.parameters():
        #     p.requires_grad = False
        # for p in self.actor_master.parameters():
        #     p.requires_grad = False

        # navigation policy
        self.actor_1_ = LinearActorHead_(self._hidden_size, action_space_num['nav'] + 1)
        self.actor_nav = LinearActorHead_(self._hidden_size, action_space_num['nav'] - 1)
        self.critic_1_ = LinearActorHead_(self._hidden_size, action_space_num['nav'])
        # for p in self.actor_nav.parameters():
        #     p.requires_grad = False
        # for p in self.actor_1_.parameters():
        #     p.requires_grad = False
        # for p in self.critic_1_.parameters():
        #     p.requires_grad = False

        # push policy
        self.actor_2 = LinearActorHead_(self._hidden_size, action_space_num['move'])
        self.critic_2 = LinearCriticHead(self._hidden_size)
        # for p in self.actor_2.parameters():
        #     p.requires_grad = False
        # for p in self.critic_2.parameters():
        #     p.requires_grad = False

        # pick policy
        self.actor_3 = LinearActorHead_(self._hidden_size, action_space_num['pick'])
        self.critic_3 = LinearCriticHead(self._hidden_size)
        # for p in self.actor_3.parameters():
        #     p.requires_grad = False
        # for p in self.critic_3.parameters():
        #     p.requires_grad = False

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.intent_embedding_all = nn.Linear(12, 12)
        self.intent_embedding_all_ = nn.Linear(12, 12)
        for p in self.intent_embedding_all_.parameters():
            p.requires_grad = False

        self.state_encoder_m = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + 12,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        # for p in self.state_encoder_m.parameters():
        #     p.requires_grad = False

        self.state_encoder_intent = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + 12,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        for p in self.state_encoder_intent.parameters():
            p.requires_grad = False

        self.actor_intent_ = LinearActorHead(self._hidden_size, action_space_num['decision'])
        for p in self.actor_intent_.parameters():
            p.requires_grad = False

        self.data_num = 0
        self.train()
        self.mode = [0] * 40
        self.last_step_size = 1

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder_.is_blind

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
            rnn_p=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            rnn_intent = (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            rnn_m = (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,  # 步数step
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]
        x_ = x
        x_p = x

        # debug
        if torch.isinf(observations['depth']).any():
            print('depth is inf!')
        if torch.isinf(observations['rgb']).any():
            print('rgb is inf!')
        observations['rgb'] = torch.where(torch.isinf(observations['rgb']), torch.full_like(observations['rgb'], 1), observations['rgb'])
        observations['depth'] = torch.where(torch.isinf(observations['depth']), torch.full_like(observations['depth'], 1),
                                          observations['depth'])
        observations['rgb'] = torch.where(torch.isnan(observations['rgb']), torch.full_like(observations['rgb'], 1),
                                          observations['rgb'])
        observations['depth'] = torch.where(torch.isnan(observations['depth']),
                                            torch.full_like(observations['depth'], 1),
                                            observations['depth'])

        # synchronize the intent policy
        if self.last_step_size == 30:
            self.state_encoder_intent.load_state_dict(self.state_encoder_m.state_dict())
            self.actor_intent_.load_state_dict(self.actor_master.state_dict())
            self.intent_embedding_all_.load_state_dict(self.intent_embedding_all.state_dict())

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)

            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        if not self.is_blind:
            perception_embed_ = self.visual_encoder_(observations)
            if self.sensor_fusion:
                perception_embed_ = self.sensor_fuser_(perception_embed_)

            x_ = [perception_embed_] + x_

        x_ = torch.cat(x_, dim=-1)
        x_master = x_
        device_ = x.device
        x_, rnn_hidden_states_ = self.state_encoder_(x_, memory.tensor("rnn_"), masks)

        # pick policy
        if not self.is_blind:
            perception_embed_p = self.visual_encoder_pick(observations)
            if self.sensor_fusion:
                perception_embed_p = self.sensor_fuser_pick(perception_embed_p)

            x_p = [perception_embed_p] + x_p

        x_p = torch.cat(x_p, dim=-1)
        x_p, rnn_hidden_states_p = self.state_encoder_pick(x_p, memory.tensor("rnn_p"), masks)

        void_intent = torch.full((x.shape[0], x.shape[1], 12), 1 / 12).to(device_)
        void_intent = self.intent_embedding_all_(void_intent)
        x_intent = torch.cat((void_intent, x_master), dim=2)

        x_intent, rnn_hidden_states_intent = self.state_encoder_intent(x_intent, memory.tensor("rnn_intent"), masks)
        intent = self.actor_intent_(x_intent)
        intent = torch.softmax(intent, dim=-1)
        # simulate the low-level action decision for intent integration
        action_intent = torch.zeros([x.shape[0], x.shape[1], self.action_space.n]).to(device_)
        move_action = torch.zeros([x.shape[0], x.shape[1], action_space_num['move']]).to(device_)
        pick_action = torch.zeros([x.shape[0], x.shape[1], action_space_num['pick']]).to(device_)
        nav_action = torch.zeros([x.shape[0], x.shape[1], action_space_num['nav']-1]).to(device_)
        for sample in range(x.shape[1]):
            for step in range(x.shape[0]):
                # record the mode
                if prev_actions[0][sample] in [5, 6, 7, 8]:
                    self.mode[sample] = 1
                if prev_actions[0][sample] in [9, 10]:
                    self.mode[sample] = 2
                if self.mode[sample] != 0 and prev_actions[0][sample] == 11:
                    self.mode[sample] = 0

                ind = [i + 5 for i in range(action_space_num['move'] - 1)]
                index = (torch.LongTensor([step for i in range(action_space_num['move'] - 1)]).to(device_),
                         torch.LongTensor([sample for i in range(action_space_num['move'] - 1)]).to(device_),
                         torch.LongTensor(ind).to(device_))
                # push
                move_action[step,sample,:] = self.actor_2(x[step][sample].view(1, 1, -1)).view(-1)
                move_action_ = move_action[step,sample,:4]
                if self.mode[sample] == 0:
                    action_intent.index_put_(index, torch.softmax(move_action_, dim=-1) * intent[step][sample][1])
                elif self.mode[sample] == 1:
                    ind.extend([j for j in range(11, 12)])
                    index = (torch.LongTensor([step for i in range(action_space_num['move'])]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['move'])]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    action_intent.index_put_(index, torch.softmax(move_action[step,sample,:], dim=-1))

                ind = [i for i in range(1, 3)]
                ind.extend([j for j in range(9, 11)])
                index = (torch.LongTensor([step for i in range(action_space_num['pick'] - 1)]).to(device_),
                         torch.LongTensor([sample for i in range(action_space_num['pick'] - 1)]).to(device_),
                         torch.LongTensor(ind).to(device_))
                # pick
                pick_action[step,sample,:]= self.actor_3(x_p[step][sample].view(1, 1, -1)).view(-1)
                pick_action_ = pick_action[step,sample,:4]
                if self.mode[sample] == 0:
                    action_intent.index_put_(index, torch.softmax(pick_action_, dim=-1) * intent[step][sample][2])
                elif self.mode[sample] == 2:
                    ind.extend([j for j in range(11, 12)])
                    index = (torch.LongTensor([step for i in range(action_space_num['pick'])]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['pick'])]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    action_intent.index_put_(index, torch.softmax(pick_action[step,sample,:], dim=-1))

                if self.mode[sample] == 0:
                    ind = [i for i in range(action_space_num['nav']-2)]
                    index = (torch.LongTensor([step for i in range(action_space_num['nav']-2)]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['nav']-2)]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    # nav
                    nav_action[step,sample,:] = self.actor_nav(x_[step][sample].view(1, 1, -1)).view(-1)
                    action_intent.index_put_(index, torch.softmax(nav_action[step,sample,:], dim=-1)[:3] * intent[step][sample][0])

                    # END
                    index = (torch.LongTensor([step]).to(device_),
                             torch.LongTensor([sample]).to(device_),
                             torch.LongTensor([self.action_space.n - 1]).to(device_))
                    action_intent.index_put_(index, intent[step][sample][3])

        action_intent = self.intent_embedding_all(action_intent)

        x_master = torch.cat((action_intent, x_master), dim=2)
        x_master, rnn_hidden_states_m = self.state_encoder_m(x_master, memory.tensor("rnn_m"), masks)
        cat = self.actor_master(x_master)

        critic = self.critic(x_master)
        decision = torch.softmax(cat, dim=-1)

        action = torch.zeros([x.shape[0], x.shape[1], self.action_space.n]).to(device_)
        for sample in range(x.shape[1]):
            for step in range(x.shape[0]):
                ind = [i + 5 for i in range(action_space_num['move'] - 1)]
                index = (torch.LongTensor([step for i in range(action_space_num['move'] - 1)]).to(device_),
                         torch.LongTensor([sample for i in range(action_space_num['move'] - 1)]).to(device_),
                         torch.LongTensor(ind).to(device_))
                # push
                if self.mode[sample] == 0:
                    action.index_put_(index, torch.softmax(move_action[step,sample,:4], dim=-1) * decision[step][sample][1])
                elif self.mode[sample] == 1:
                    ind.extend([j for j in range(11, 12)])
                    index = (torch.LongTensor([step for i in range(action_space_num['move'])]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['move'])]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    action.index_put_(index, torch.softmax(move_action[step,sample,:], dim=-1))

                ind = [i for i in range(1, 3)]
                ind.extend([j for j in range(9, 11)])
                index = (torch.LongTensor([step for i in range(action_space_num['pick'] - 1)]).to(device_),
                         torch.LongTensor([sample for i in range(action_space_num['pick'] - 1)]).to(device_),
                         torch.LongTensor(ind).to(device_))
                # pick
                if self.mode[sample] == 0:
                    action.index_put_(index, torch.softmax(pick_action[step,sample,:4], dim=-1) * decision[step][sample][2])
                elif self.mode[sample] == 2:
                    ind.extend([j for j in range(11, 12)])
                    index = (torch.LongTensor([step for i in range(action_space_num['pick'])]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['pick'])]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    action.index_put_(index, torch.softmax(pick_action[step,sample,:], dim=-1))

                if self.mode[sample] == 0:
                    ind = [i for i in range(action_space_num['nav']-2)]
                    index = (torch.LongTensor([step for i in range(action_space_num['nav']-2)]).to(device_),
                             torch.LongTensor([sample for i in range(action_space_num['nav']-2)]).to(device_),
                             torch.LongTensor(ind).to(device_))
                    # nav
                    action.index_put_(index,
                                      torch.softmax(nav_action[step,sample,:],
                                                    dim=-1)[:3] * decision[step][sample][0])

                    # END
                    index = (torch.LongTensor([step]).to(device_),
                             torch.LongTensor([sample]).to(device_),
                             torch.LongTensor([self.action_space.n - 1]).to(device_))
                    action.index_put_(index, decision[step][sample][3])

        action = CategoricalDistr(probs=action)

        ac_output = ActorCriticOutput(
            distributions=action, values=critic, extras={}
        )
        memory.set_tensor("rnn", rnn_hidden_states)
        memory.set_tensor("rnn_", rnn_hidden_states_)
        memory.set_tensor("rnn_p", rnn_hidden_states_p)
        memory.set_tensor("rnn_intent", rnn_hidden_states_intent)
        memory.set_tensor("rnn_m", rnn_hidden_states_m)

        self.last_step_size = x.shape[0]

        return ac_output, memory