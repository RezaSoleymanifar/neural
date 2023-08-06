from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class StableBaselinesActorCriticPolicy(ActorCriticPolicy):

    def set_policy_value_networks(cls, policy_network: nn.Module, value_network: nn.Module):
        """
        Set the policy and value networks for the policy.
        """
        cls.policy_network = policy_network
        cls.value_network = value_network
        return None

    def build_policy_value_networks(cls, features_dim):
        """
        Build the policy and value networks using the specified features dimension.
        """
        class CustomNetwork(nn.Module):

            def __init__(
                self,
                features_dim: Tuple[int],
                policy_network: Type[nn.Module],
                value_network: Type[nn.Module],
            ):
                self.policy_network = policy_network(features_dim)
                self.value_network = value_network(features_dim)
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
            
        return CustomNetwork(features_dim, cls.policy_network, cls.value_network)

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
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self.build_policy_value_networks(self.features_dim)
