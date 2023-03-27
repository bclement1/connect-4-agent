"""  
Exploration de la librairie Petting Zoo et de l'environnenement du Connect-Four.
"""

from pettingzoo.classic import connect_four_v3
import matplotlib.pyplot as plt

env = connect_four_v3.env(render_mode="human")


def policy(obs, agent, env):
    """
    Random policy.
    """
    return env.action_space(agent).sample()


env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation, agent, env)
    print(f"Agent: {agent} Action: {action}")
    env.step(action)
env.render()
