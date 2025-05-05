import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode
import torch.nn.functional as F
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



class NeuralNetwork(nn.Module):
    """
    This class implements a neural network with a variable number of hidden layers and hidden units.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        hidden_layers: int,
        activation: str,
    ):
        super(NeuralNetwork, self).__init__()

        activation_fn = nn.ReLU()

        self.model = nn.Sequential()
        self.model.add_module("dense1", nn.Linear(input_dim, hidden_size))
        self.model.add_module("activation1", activation_fn)
        for i in range(hidden_layers):
            self.model.add_module(
                "dense" + str(i + 2), nn.Linear(hidden_size, hidden_size)
            )
            self.model.add_module("activation" + str(i + 2), activation_fn)
        self.model.add_module(
            "dense" + str(hidden_layers + 2), nn.Linear(hidden_size, output_dim)
        )
        # model.add_module("softmax", nn.Tanh())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)


class Actor:
    """
    This class implements the actor in the SAC algorithm for RL
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        actor_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        print("Actor on device: {}".format(self.device))
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()
        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr)

    def setup_actor(self):
        """
        This function sets up the actor network in the Actor class.
        """
        self.actor_network = NeuralNetwork(
            input_dim=self.state_dim,
            output_dim=2,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            activation="relu",
        )
        self.actor_network.to(self.device)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        """
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        """
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        """

        assert (
            state.shape == (3,) or state.shape[1] == self.state_dim
        ), "State passed to this method has a wrong shape"

        forward_pass = self.actor_network(state)
        try:
            mean, log_std = forward_pass[:, 0], forward_pass[:, 1]
        except:
            mean, log_std = forward_pass[0], forward_pass[1]
        log_std = self.clamp_log_std(log_std)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            normal = Normal(mean, std)
            normal_log_prob = normal.log_prob(action)
            subbend = torch.log(1 - action**2 + 0.001)

            log_prob = normal_log_prob - subbend

        else:
            normal = Normal(mean, std)
            sample = normal.rsample()
            action = torch.tanh(sample)
            normal_log_prob = normal.log_prob(sample)
            subbend = torch.log(1 - action**2 + 0.001)
            log_prob = normal_log_prob - subbend

        action = action.squeeze().unsqueeze(-1)
        log_prob = log_prob.squeeze().unsqueeze(-1)

        assert (
            action.shape == (self.action_dim,) and log_prob.shape == (self.action_dim,)
        ) or (
            action.shape == (state.shape[0], 1)
            and log_prob.shape == (state.shape[0], 1)
        ), "Incorrect shape for action or log_prob."
        return action, log_prob


class Critic:
    """
    This class implements the Critic in the SAC algorithm for RL, using a network for the Q value and one for a target Q
    """
    def __init__(
        self,
        hidden_size: int,
        hidden_layers: int,
        critic_lr: float,
        state_dim: int = 3,
        action_dim: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        print("Critic on device: {}".format(self.device))
        self.setup_critic()
        self.optimizer = optim.Adam(list(self.qnetwork.parameters()), lr=self.critic_lr)

    def setup_critic(self):
        self.qnetwork = NeuralNetwork(
            input_dim=self.state_dim + self.action_dim,
            output_dim=1,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            activation="relu",
        )
        self.qnetwork.to(self.device)
        self.q_target_network = NeuralNetwork(
            input_dim=self.state_dim + self.action_dim,
            output_dim=1,
            hidden_size=self.hidden_size,
            hidden_layers=self.hidden_layers,
            activation="relu",
        )
        self.q_target_network.to(self.device)


class TrainableParameter:
    """
    This class could be used to define a trainable parameter. Could be useful for the entropy temperature parameter
    for SAC algorithm.
    """

    def __init__(
        self,
        init_param: float,
        lr_param: float,
        train_param: bool,
        device: torch.device = torch.device("cpu"),
    ):
        self.log_param = torch.tensor(
            np.log(init_param), requires_grad=train_param, device=device
        )
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    """
    Class to implement the agent that applies force to the swinging pendulum
    """
    def __init__(self):
        # Environment variables
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(
            self.min_buffer_size, self.max_buffer_size, self.device
        )
        self.gamma = 0.99
        self.tau = 0.005
        self.global_step = 0
        self.target_update_interval = 1
        self.unique_lr = 1e-4
        self.alpha_lr = 3e-5
        self.alpha = TrainableParameter(
            0.1, lr_param=self.alpha_lr, train_param=True, device=self.device
        )
        self.target_temp = -1
        self.soft_update = True

        self.setup_agent()

    def setup_agent(self):
        """
        Setup off-policy agent with policy and critic classes.
        """
        lr = self.unique_lr
        hidden = 256
        layers = 2
        self.actor = Actor(
            hidden_size=hidden, hidden_layers=layers, actor_lr=lr, device=self.device
        )
        self.critic_1 = Critic(
            hidden_size=hidden, hidden_layers=layers, critic_lr=lr, device=self.device
        )
        self.critic_2 = Critic(
            hidden_size=hidden, hidden_layers=layers, critic_lr=lr, device=self.device
        )

        self.critic_target_update(
            self.critic_1.qnetwork, self.critic_1.q_target_network, self.tau, False
        )
        self.critic_target_update(
            self.critic_2.qnetwork, self.critic_2.q_target_network, self.tau, False
        )

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        Returns an action from the policy for state s
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        state = torch.Tensor(s).to(self.device)
        forward = self.actor.actor_network(state)
        mean, log_std = forward.split(self.action_dim)

        log_std = self.actor.clamp_log_std(log_std)
        std = log_std.exp()

        probabilities = Normal(mean, std)
        actions = probabilities.rsample()
        action = torch.tanh(actions)
        action = action.cpu().detach().numpy()

        assert action.shape == (1,), "Incorrect action shape."
        assert isinstance(action, np.ndarray), "Action dtype must be np.ndarray"
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        """
        This function takes in an object containing trainable parameters and an optimizer,
        and using a given loss, runs one step of gradient update.
        :param object: object containing trainable parameters and an optimizer
        """
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(
        self,
        base_net: NeuralNetwork,
        target_net: NeuralNetwork,
        tau: float,
        soft_update: bool,
    ):
        """
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        """
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(
                    param_target.data * (1.0 - tau) + param.data * tau
                )
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        """
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer,and then updates the policy and critic networks
        using the sampled batch.
        """
        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        # Critic loss
        with torch.no_grad():
            alpha = self.alpha.get_param()
            next_state_actions, next_log_probs = self.actor.get_action_and_log_prob(
                s_prime_batch, deterministic=False
            )
            new_input = torch.hstack((s_prime_batch, next_state_actions))
            q1_next = self.critic_1.q_target_network(new_input)
            q2_next = self.critic_2.q_target_network(new_input)
            min_q = torch.min(q1_next, q2_next) - alpha * next_log_probs
            next_q_value = r_batch.flatten() + self.gamma * min_q.view(-1)

        old_input = torch.hstack((s_batch, a_batch))
        q1_a_values = self.critic_1.qnetwork(old_input).view(-1)
        q2_a_values = self.critic_2.qnetwork(old_input).view(-1)
        q1_loss = F.mse_loss(q1_a_values, next_q_value)
        q2_loss = F.mse_loss(q2_a_values, next_q_value)

        # Critic update
        self.run_gradient_update_step(self.critic_1, q1_loss)
        self.run_gradient_update_step(self.critic_2, q2_loss)

        # Policy loss
        state_actions, log_probs = self.actor.get_action_and_log_prob(
            s_batch, deterministic=False
        )
        policy_input = torch.hstack((s_batch, state_actions))
        q1_pi = self.critic_1.qnetwork(policy_input)
        q2_pi = self.critic_2.qnetwork(policy_input)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (alpha * log_probs) - min_q_pi

        # Actor update
        self.run_gradient_update_step(self.actor, policy_loss)

        with torch.no_grad():
            state_actions, log_probs = self.actor.get_action_and_log_prob(
                s_batch, deterministic=True
            )

        # update alpha
        alpha = self.alpha.get_param()
        alpha_loss = -alpha * (log_probs + self.target_temp)

        self.run_gradient_update_step(self.alpha, alpha_loss)

        self.global_step += 1

        # Target update
        if self.global_step % self.target_update_interval == 0:
            with torch.no_grad():
                self.critic_target_update(
                    self.critic_1.qnetwork,
                    self.critic_1.q_target_network,
                    self.tau,
                    self.soft_update,
                )
                self.critic_target_update(
                    self.critic_2.qnetwork,
                    self.critic_2.q_target_network,
                    self.tau,
                    self.soft_update,
                )


# This main function is provided here to enable some basic testing.
if __name__ == "__main__":
    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print("\n")

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
