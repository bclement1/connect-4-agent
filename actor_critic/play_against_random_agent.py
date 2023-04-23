import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import connect_four_v3
from connect4_a2c import PlayerA2C
from IPython.display import clear_output

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(2)


class RandomPlayer:
    def __init__(self, env, player_id):
        self.wins = 0
        self.defeats = 0
        self.env = env
        self.player_id = player_id

    def get_action(self):
        state_dict, _, truncated, terminated, _ = self.env.last()
        action_mask = state_dict["action_mask"]
        if truncated or terminated:
            action = None
        else:
            action = self.env.action_spaces["player_" + str(self.player_id)].sample()
            while not action_mask[action]:
                action = self.env.action_spaces[
                    "player_" + str(self.player_id)
                ].sample()
        return action

    def count_points(self, reward):
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.defeats += 1

    def play(self):
        action = self.get_action()
        self.env.step(action)
        _, reward, truncated, terminated, _ = self.env.last()
        if truncated or terminated:
            self.count_points(reward)

    def __str__(self):
        return "Random"


NB_GAMES = 3000


def sort_agents_by_player_id(agents: list):
    """Sort the agents list so that the first player is at the first position and so on."""
    agents_sorted = [None, None]
    for agent in agents:
        agents_sorted[agent.player_id] = agent
    return agents_sorted


def train_a2c_against_randy(display=False, a2c_heuristic=False):
    """Make A2C play against a random agent."""
    env = connect_four_v3.env(render_mode="rgb_array")
    env.reset()
    # Lists to store wins
    a2c_wins = []
    random_wins = []
    agents = [
        PlayerA2C(env=env, player_id=0, heuristic=a2c_heuristic),
        RandomPlayer(env=env, player_id=1),
    ]
    for i in range(NB_GAMES):
        logging.info("Game number: %d", i)
        # Indicate whether a game is finished or not
        truncated = False
        terminated = False
        # Draw at random the players order
        player_order = np.random.choice([0, 1], size=2, replace=False)
        for idx, agent in enumerate(agents):
            agent.player_id = player_order[idx]
        agents = sort_agents_by_player_id(agents)
        player_id = 0
        turns_counter = 1
        print("First player ", agents[0])
        while not (truncated or terminated):
            logging.debug("Turn %d", turns_counter)
            # Player plays
            logging.debug("Player %d plays", player_id)
            agents[player_id].play()
            player_id += 1
            player_id = player_id % 2
            turns_counter += 1
            _, _, truncated, terminated, _ = env.last()
            if display:
                clear_output(wait=True)
                plt.imshow(env.render())
                plt.show()
            if truncated or terminated:
                # The last agent plays one more time
                # to take the defeat or victory into account
                agents[player_id].play()
                env.reset()
                for agent in agents:
                    if isinstance(agent, RandomPlayer):
                        random_wins.append(agent.wins)
                    elif isinstance(agent, PlayerA2C):
                        a2c_wins.append(agent.wins)
    for agent in agents:
        if isinstance(agent, PlayerA2C):
            cumulated_rewards_list = agent.cumulated_rewards_list

    print("A2C total wins: ", a2c_wins[-1])
    print("RANDOM total wins: ", random_wins[-1])
    print("Number of draws: ", NB_GAMES - (random_wins[-1] + a2c_wins[-1]))

    nb_games = [1 + i for i in range(NB_GAMES)]
    plt.plot(nb_games, a2c_wins, label="Victories a2c")
    plt.plot(nb_games, random_wins, label="Victories of the random agent")
    plt.xlabel("Number of games")
    plt.ylabel("Number of victories")
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(nb_games, cumulated_rewards_list, label="Cumulated rewards")
    plt.plot(
        nb_games,
        [
            np.mean(cumulated_rewards_list[: i + 1])
            for i in range(len(cumulated_rewards_list))
        ],
        label="Cumulated rewards on average",
    )
    plt.ylabel("Reward")
    plt.xlabel("Number of games")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_a2c_against_randy(display=False)
