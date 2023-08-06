from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy


class StableBaselinesActorCriticPolicy(ActorCriticPolicy):

    def set_actor_critic_networks(cls, actor_network: nn.Module, critic_network: nn.Module):
        """
        Set the policy and value networks for the policy.
        """
        cls.actor_network = actor_network
        cls.critic_network = critic_network
        return None

    def build_actor_critic_policy(cls, features_dim):
        """
        Build the policy and value networks using the specified features dimension.
        """
        class CustomActorCriticPolicy(nn.Module):

            def __init__(
                self,
                features_dim: Tuple[int],
                actor_network: Type[nn.Module],
                critic_network: Type[nn.Module],
            ):
                self.actor_network = actor_network(features_dim)
                self.critic_network = critic_network(features_dim)
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
            
        return CustomActorCriticPolicy(features_dim, cls.actor_network, cls.critic_network)

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
        self.mlp_extractor = self.build_actor_critic_policy(self.features_dim)
