import numpy
import torch
import torch.nn.functional as F

from agent_models import ACModel
from torch_rl.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, envs, acmodel:ACModel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, num_recurr_steps)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        inds = numpy.arange(0, self.num_frames, self.num_recurr_steps)
        self.acmodel.set_hidden_state(exps[inds].agent_steps)
        for i in range(self.num_recurr_steps):
            sb = exps[inds + i]
            dist, value, _ = self.acmodel(sb.env_steps)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.agent_steps.actions) * sb.advantages).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.num_recurr_steps
        update_value /= self.num_recurr_steps
        update_policy_loss /= self.num_recurr_steps
        update_value_loss /= self.num_recurr_steps
        update_loss /= self.num_recurr_steps

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs["entropy"] = update_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.num_recurr_steps)
        return starting_indexes
