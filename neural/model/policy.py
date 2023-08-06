
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
On-Policy Algorithms
Custom Networks
If you need a network architecture that is different for the actor and the critic when using PPO, A2C or TRPO, you can pass a dictionary of the following structure: dict(pi=[<actor network architecture>], vf=[<critic network architecture>]).

For example, if you want a different architecture for the actor (aka pi) and the critic ( value-function aka vf) networks, then you can specify net_arch=dict(pi=[32, 32], vf=[64, 64]).

Otherwise, to have actor and critic that share the same network architecture, you only need to specify net_arch=[128, 128] (here, two hidden layers of 128 units each, this is equivalent to net_arch=dict(pi=[128, 128], vf=[128, 128])).

If shared layers are needed, you need to implement a custom policy network (see advanced example below).

Examples
Same architecture for actor and critic with two layers of size 128: net_arch=[128, 128]

        obs
   /            \
 <128>          <128>
  |              |
 <128>          <128>
  |              |
action         value
Different architectures for actor and critic: net_arch=dict(pi=[32, 32], vf=[64, 64])

        obs
   /            \
 <32>          <64>
  |              |
 <32>          <64>
  |              |
action         value
Advanced Example
If your task requires even more granular control over the policy/value architecture, you can redefine the policy directly:

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):

    def __init__(
        self,
        policy_network: Type[nn.Module],
        value_network: Type[nn.Module],
    ):
        self.policy_network = policy_network
        self.value_network = value_network
        super().__init__()

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)
