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
import tqdm
from pettingzoo.classic import connect_four_v3
import numpy as np
import random
import json
import os

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
    next_position[np.max(np.where(next_position[:, action].sum(axis=1) == 0)[0])][action][0] = 1
    return next_position


# ******************************************************************************************************************** #
# Actor definition
class Neuronal_Agent:
    def __init__(
        self,
        player,
        random_proportion=0.3,
        num_epochs=1000,
        learning_rate=0.01,
        name=None,
        optimizer="Adam",
        loss="MeanAbsoluteError",
    ):
        self.network = keras.Sequential(
            [
                layers.Conv2D(
                    kernel_size=2, filters=2, padding="same", strides=(1, 1), activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),
                layers.Conv2D(
                    kernel_size=2, filters=2, padding="same", strides=(1, 1), activation="relu",
                ),
                tf.keras.layers.BatchNormalization(),

                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(4),
                layers.Dense(1),
            ]
        )
        self.player = player
        self.random_proportion = random_proportion
        self.name = name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss

    def update_random_proportion(self, random_proportion):
        self.random_proportion = random_proportion

    def get_value(self, observation):
        actions = possibles_actions(observation)
        random_indicateur = random.random()
        if random_indicateur < self.random_proportion:
            return random.choice(actions)
        else:
            position = np.copy(observation[0]["observation"]).astype(float)
            position = np.expand_dims(position, axis=0)
            new_score = -np.inf
            for action in actions:
                next_position = apply_action(position, action, self.player)
                next_position = np.expand_dims(next_position, axis=0)
                next_position_score = self.network(next_position)
                if next_position_score > new_score:
                    next_action = action
                    new_score = next_position_score
            return next_action

    def save(self, path):
        os.makedirs(path,exist_ok=True)
        params = {}
        params["player"] = self.player
        params["random_proportion"] = self.random_proportion
        params["name"] = self.name
        params["num_epochs"] = self.num_epochs
        params["learning_rate"] = self.learning_rate
        params["optimizer"] = self.optimizer
        params["loss"] = self.loss
        with open(path+"params.json", "w") as write_file:
            json.dump(params, write_file, indent=4)
        self.network.save(path+"model.model")

    def load(self, path):
        self.network = tf.keras.models.load_model(path+"model.model")
        file = open(path+"params.json")
        params = json.load(file)
        self.player = params.get("player")
        self.loss = params.get("loss")
        self.optimizer = params.get("optimizer")
        self.learning_rate = params.get("learning_rate")
        self.num_epochs = params.get("num_epochs")
        self.random_proportion = params.get("random_proportion")



# ******************************************************************************************************************** #
# Configuration
num_epochs_list = [1000,3000]
learning_rate = 0.0001
random_rate = 0.3
Nb_training= 3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_MAE = tf.keras.losses.MeanAbsoluteError()
# ******************************************************************************************************************** #
# Main

if __name__ == "__main__":
    for num_epochs in num_epochs_list:
        ### Game initialization
        env_4_connect = connect_four_v3.env()
        player_0 = Neuronal_Agent(
            name="Player_0",
            random_proportion=random_rate,
            player=0,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer="Adam",
            loss="MeanAbsoluteError",
        )
        player_1 = Neuronal_Agent(
            name="Player_0",
            random_proportion=random_rate,
            player=1,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer="Adam",
            loss="MeanAbsoluteError",
        )
        player_0.network.build([1, 6, 7, 2])
        player_0.network.build([1, 6, 7, 2])
        player_1.network.compile(
            optimizer=optimizer,
            loss=loss_MAE
        )
        player_1.network.compile(
            optimizer=optimizer,
            loss=loss_MAE
        )
        win_player_0 = 0
        win_player_1 = 0

        ### Playing the game
        for i in tqdm.tqdm(range(num_epochs)):

            ### Game Reset
            env_4_connect.reset()
            positions_player_0 = []
            positions_player_1 = []
            if i % 100 == 0 and i>0 :
                player_0.update_random_proportion(random_rate/(i/100))
                player_1.update_random_proportion(random_rate/(i/100))

            ### First Observation
            observations = env_4_connect.last()

            while not (observations[2]) and not (observations[3]):

                if i % 2 == 0:
                    player_0
                    # Player 0 play
                    positions_player_0.append(observations[0]["observation"])
                    action = player_0.get_value(observations)
                    env_4_connect.step(action)


                    # Observation Player 1
                    observations = env_4_connect.last()

                    # Stock the position

                    # Player 1 play
                    positions_player_1.append(observations[0]["observation"])

                    if not (observations[2]):
                        action = player_1.get_value(observations)
                        env_4_connect.step(action)


                    # Observation Player 0
                    observations = env_4_connect.last()

                else:
                    # Player 1 play
                    positions_player_1.append(observations[0]["observation"])
                    action = player_1.get_value(observations)
                    env_4_connect.step(action)

                    # Observation Player 0
                    observations = env_4_connect.last()

                    # Player 0 play
                    positions_player_0.append(observations[0]["observation"])

                    if not (observations[2]):
                        action = player_0.get_value(observations)
                        env_4_connect.step(action)

                    # Observation Player 1
                    observations = env_4_connect.last()

            ### Affect the reward
            game_rewards = env_4_connect.rewards
            game_rewards_player_0 = np.full(len(positions_player_0), game_rewards["player_0"])
            game_rewards_player_1 = np.full(len(positions_player_1), game_rewards["player_1"])
            if game_rewards["player_0"] == 1:
                win_player_0 += 1
            if game_rewards["player_1"] == 1:
                win_player_1 += 1

            ### Rewards modulation

            game_rewards_player_0 = np.cumsum(game_rewards_player_0)
            game_rewards_player_1 = np.cumsum(game_rewards_player_1)

            ### Neural Network Improvement
            positions_player = np.concatenate([positions_player_0 ,positions_player_1])
            game_rewards_player = np.concatenate([game_rewards_player_0,game_rewards_player_1])

            for k in range(Nb_training):

                # Player 0
                for j in range(len(positions_player)):
                    position = positions_player[j].astype(float)
                    position = np.expand_dims(position, axis=0)
                    true_eval = game_rewards_player[j]
                    true_eval = np.expand_dims(true_eval, axis=0)

                    with tf.GradientTape() as tape:
                        eval = player_0.network(position)
                        loss = loss_MAE(true_eval, eval)

                    gradients = tape.gradient(loss, player_0.network.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, player_0.network.trainable_weights))

                # Player 1
                for j in range(len(positions_player)):
                    position = positions_player[j].astype(float)
                    position = np.expand_dims(position, axis=0)
                    true_eval = game_rewards_player[j]
                    true_eval = np.expand_dims(true_eval, axis=0)

                    with tf.GradientTape() as tape:
                        eval = player_1.network(position)
                        loss = loss_MAE(true_eval, eval)

                    gradients = tape.gradient(loss, player_1.network.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, player_1.network.trainable_weights))

            if i % 10 == 0:
                print()
                print("Epochs :", i)
                print("win_player_0 :", win_player_0)
                print("win_player_1 :", win_player_1)
                print()
        player_0.save("player_small_{}_3_wm/".format(num_epochs))
