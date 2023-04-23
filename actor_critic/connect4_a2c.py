"""Advantage actor critic for Connect4."""
from pettingzoo.classic import connect_four_v3
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch import optim
import torch

HIDDEN_LAYER_NUM = 128
GAMMA = 0.99
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001


def custom_reward(board: np.ndarray):
    n_rows = board.shape[0]
    n_cols = board.shape[1]

    score = 0

    # Evaluate center
    center_player_coins = np.sum(board[:, n_cols // 2, 1])
    center_opponent_coins = np.sum(board[:, n_cols // 2, 0])
    score += 6 * center_player_coins - 6 * center_opponent_coins

    # Evaluate rows
    for i in range(n_rows):
        for j in range(n_cols - 3):
            row = board[i, j : j + 4, :]
            score += evaluate_window(row)

    # Evaluate columns
    for i in range(n_rows - 3):
        for j in range(n_cols):
            col = board[i : i + 4, j, :]
            score += evaluate_window(col)

    # Evaluate diagonals
    for i in range(n_rows - 3):
        for j in range(n_cols - 3):
            diag = board[
                [n_rows - 1 - i, n_rows - 2 - i, n_rows - 3 - i, n_rows - 4 - i],
                [j, j + 1, j + 2, j + 3],
                :,
            ]
            score += evaluate_window(diag)

    for i in range(n_rows - 3):
        for j in range(n_cols - 3):
            diag = board[
                [n_rows - 1 - i, n_rows - 2 - i, n_rows - 3 - i, n_rows - 4 - i],
                [n_cols - 1 - j, n_cols - 2 - j, n_cols - 3 - j, n_cols - 4 - j],
                :,
            ]
            score += evaluate_window(diag)
    return score


def evaluate_window(window: np.ndarray):
    nb_player_coins = np.sum(window[:, 1])
    nb_opponent_coins = np.sum(window[:, 0])
    if nb_player_coins == 2 and nb_opponent_coins == 0:
        return 3
    elif nb_player_coins == 3 and nb_opponent_coins == 0:
        return 8
    elif nb_player_coins == 4:
        return 10
    elif nb_opponent_coins == 3 and nb_player_coins == 0:
        return -1
    elif nb_opponent_coins == 2 and nb_player_coins == 0:
        return -1
    elif nb_opponent_coins == 4:
        return -1
    elif nb_player_coins == 1 and nb_opponent_coins == 0:
        return 1
    else:
        return 0


class ActorNetwork(nn.Module):
    def __init__(self, obs_nb_inputs: int, action_nb_outputs: int):
        super().__init__()
        self.input_layer = nn.Linear(obs_nb_inputs, HIDDEN_LAYER_NUM)
        self.hidden_layer = nn.Linear(HIDDEN_LAYER_NUM, HIDDEN_LAYER_NUM)
        self.output_layer = nn.Linear(HIDDEN_LAYER_NUM, action_nb_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return Categorical(x)


class CriticNetwork(nn.Module):
    def __init__(self, obs_nb_inputs: int):
        super().__init__()
        self.input_layer = nn.Linear(obs_nb_inputs, HIDDEN_LAYER_NUM)
        self.hidden_layer = nn.Linear(HIDDEN_LAYER_NUM, HIDDEN_LAYER_NUM)
        self.output_layer = nn.Linear(HIDDEN_LAYER_NUM, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class PlayerA2C:
    def __init__(self, env, player_id: int, heuristic: bool = True):
        self.player_id = player_id
        self.env = env
        self.heuristic = heuristic
        # Counters to store the number of wins and defeats
        self.wins = 0
        self.defeats = 0
        # List to store cumulated reward
        self.cumulated_rewards_list = []
        self.cumulated_reward = 0
        # Store the last action
        self.last_action = torch.Tensor([-1])  # No action
        (dim1, dim2, dim3) = self.env.observation_space("player_" + str(player_id))[
            "observation"
        ].shape
        obs_nb_inputs = dim1 * dim2 * dim3
        nb_actions = self.env.action_space("player_" + str(player_id)).n
        self.nb_actions = nb_actions
        self.actor = ActorNetwork(
            obs_nb_inputs=obs_nb_inputs, action_nb_outputs=self.nb_actions
        )
        self.critic = CriticNetwork(obs_nb_inputs=obs_nb_inputs)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters())
        self.actor_optimizer = optim.Adam(params=self.critic.parameters())
        self.critic_fn_loss = nn.MSELoss()

    def get_action(self, state):
        current_state_dict, _, truncated, terminated, _ = self.env.last()
        action_mask = current_state_dict["action_mask"]
        if truncated or terminated:
            action_dist = self.actor(torch.Tensor(state).flatten())
            action_of_net = self.last_action
            action_for_env = None
        else:
            action_dist: Categorical = self.actor(torch.Tensor(state).flatten())
            action_of_net = action_dist.sample()
            action_for_env = int(action_of_net)
            while not action_mask[action_for_env]:
                action_of_net = action_dist.sample()
                action_for_env = int(action_of_net)
        return action_of_net, action_for_env, action_dist

    def count_points(self, reward):
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.defeats += 1

    def play(self):
        current_state = self.env.last()[0]["observation"]
        value_state = self.critic(torch.Tensor(current_state).flatten())
        action_of_net, action_for_env, action_dist = self.get_action(current_state)
        self.last_action = action_of_net
        # Act
        self.env.step(action_for_env)
        # Observe action and reward
        next_state_dict, reward, truncated, terminated, _ = self.env.last()
        value_next_state = self.critic(
            torch.Tensor(next_state_dict["observation"]).flatten()
        )
        heuristic_reward = custom_reward(next_state_dict["observation"])
        if not (truncated or terminated):
            if self.heuristic:
                target = heuristic_reward + GAMMA * value_next_state.detach()
                self.cumulated_reward += heuristic_reward
            else:
                target = reward + GAMMA * value_next_state.detach()
                self.cumulated_reward += reward

        elif truncated or terminated:
            if self.heuristic:
                target = torch.Tensor([heuristic_reward])
                self.cumulated_reward += heuristic_reward
                self.cumulated_rewards_list.append(self.cumulated_reward)
                self.cumulated_reward = 0
            else:
                target = torch.Tensor([reward])
                self.cumulated_reward += reward
                self.cumulated_rewards_list.append(self.cumulated_reward)
                self.cumulated_reward = 0
            self.count_points(reward)
        # Calculate losses
        critic_loss = self.critic_fn_loss(value_state, target)
        actor_loss = (
            -action_dist.log_prob(action_of_net).unsqueeze(0) * critic_loss.detach()
        )
        # Backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def __str__(self):
        return "A2C"
