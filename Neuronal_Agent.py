# ******************************************************************************************************************** #
# ********************************************* #   CENTRALESUPELEC   # ********************************************** #
# ******************************************************************************************************************** #
# Project : connect-4-agent
# File    : Neuronal_Agent.py
# PATH    :
# Author  : trisr
# Date    : 28/03/2023
# Description :
"""




"""
# Last commit ID   :
# Last commit date :
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #
# ******************************************************************************************************************** #


# ******************************************************************************************************************** #
# Importations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pettingzoo.classic import connect_four_v3
import numpy as np
import random

# ******************************************************************************************************************** #
# Functions definition


def possibles_actions(observation):
    """

    Parameters
    ----------
    positions

    Returns
    -------

    """
    actions = np.where(observation[0]["action_mask"] == 1)[0]
    return actions


def apply_action(position, action, player):
    """

    Parameters
    ----------
    position
    action

    Returns
    -------

    """
    next_position = np.copy(position[0])
    next_position[np.max(np.where(next_position[:, action].sum(axis=1) == 0)[0])][action][player] = 1
    return next_position


# ******************************************************************************************************************** #
# Actor definition
class Neuronal_Agent:
    def __init__(self, num_actions, random_proportion, player):
        self.network = keras.Sequential(
                    [
                        layers.Conv2D(
                            kernel_size=2,
                            filters=2,
                            padding='same',
                            strides=(1, 1),
                            activation="relu",
                            name="Conv1"
                        ),
                        layers.Conv2D(
                            kernel_size=2,
                            filters=2,
                            padding='same',
                            strides=(1, 1),
                            activation="relu",
                            name="Conv2"
                        ),
                        layers.MaxPool2D(),
                        layers.Flatten(),
                        layers.Dense(4, activation="relu", name="dense1"),
                        layers.Dense(1, name="dense2"),
                    ]
                )
        self.random_proportion = random_proportion
        self.player = player

    def update_random_proportion(self,random_proportion):
        self.random_proportion = random_proportion

    def get_value(self, observation):
        actions = possibles_actions(observation)
        random_indicateur = random.random()
        if random_indicateur < self.random_proportion:
            return random.choice(actions)
        else:
            position = np.copy(observation[0]["observation"]).astype(float)
            position = np.expand_dims(position, axis=0)
            position_score = self.network(position)
            new_score = -np.inf
            for action in actions:
                next_position = apply_action(position, action, self.player)
                next_position =  np.expand_dims(next_position, axis=0)
                next_position_score = self.network(next_position)
                if (next_position_score - position_score) > new_score:
                    next_action = action
                    new_score = next_position_score - position_score
            return next_action

# ******************************************************************************************************************** #
# Configuration
num_epochs = 100
learning_rate = 0.01
random_rate = 0.3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_MAE = tf.keras.losses.MeanAbsoluteError()
# ******************************************************************************************************************** #
# Main
if __name__ == "__main__":
    ### Game initialization
    env_4_connect = connect_four_v3.env()
    player_0 = Neuronal_Agent(num_actions=7,
                              random_proportion=random_rate,
                              player=0)
    player_1 = Neuronal_Agent(num_actions=7,
                              random_proportion=random_rate,
                              player=1)

    player_0.network.build([1, 6, 7, 2])
    player_1.network.build([1, 6, 7, 2])
    win_player_0 = 0
    win_player_1 = 0

    ### Playing the game
    for i in range(num_epochs):
        ### Game Reset
        env_4_connect.reset()
        positions_player_0 = []
        positions_player_1 = []

        ### First Observation
        observations = env_4_connect.last()
        while not(observations[2]) and not(observations[3]):
            # Player 0 play
            action = player_0.get_value(observations)
            env_4_connect.step(action)

            # Observation Player 1
            observations = env_4_connect.last()

            # Stock the position
            positions_player_0.append(observations[0]["observation"])

            # Player 1 play
            if not(observations[2]):
                action = player_1.get_value(observations)
                env_4_connect.step(action)

            # Observation Player 0
            observations = env_4_connect.last()

            # Stock the position
            positions_player_1.append(observations[0]["observation"])

        ### Affect the reward
        game_rewards = env_4_connect.rewards
        game_rewards_player_0 = np.full(len(positions_player_0),game_rewards["player_0"])
        game_rewards_player_1 = np.full(len(positions_player_1),game_rewards["player_1"])
        if game_rewards["player_0"] == 1:
            win_player_0 += 1
        if game_rewards["player_1"] == 1:
            win_player_1 += 1

        ### Rewards modulation



        ### Neural Network Improvement

        # Player 0
        for j in range(len(positions_player_0)):
            position = positions_player_0[j].astype(float)
            position = np.expand_dims(position, axis=0)
            true_eval = game_rewards_player_0[j]
            true_eval = np.expand_dims(true_eval, axis=0)

            with tf.GradientTape() as tape:
                eval = player_0.network(position)
                loss = loss_MAE(true_eval, eval)

            gradients = tape.gradient(loss, player_0.network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, player_0.network.trainable_weights))

        # Player 1
        for j in range(len(positions_player_1)):
            position = positions_player_1[j].astype(float)
            position = np.expand_dims(position, axis=0)
            true_eval = game_rewards_player_1[j]
            true_eval = np.expand_dims(true_eval, axis=0)

            with tf.GradientTape() as tape:
                eval = player_1.network(position)
                loss = loss_MAE(true_eval, eval)

            gradients = tape.gradient(loss, player_1.network.trainable_weights)
            optimizer.apply_gradients(zip(gradients, player_1.network.trainable_weights))


        if i%10 == 0 :
            print("Epochs :",i)
            print("win_player_0 :",win_player_0)
            print("win_player_1 :",win_player_1)
            print()