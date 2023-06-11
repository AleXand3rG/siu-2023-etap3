import numpy as np
import turtlesim_env_multi
from dqn_multi import DqnMulti
from keras.models import load_model

# Setup simulation
MOVES = 1000
NUMBER_OF_AGENTS = 8
ROUTES_FILENAME = '/root/siu/routes_multi.csv'
MODEL_FILEPATH = '/root/siu/models/model_multi.h5'

DETECT_COLLISIONS = True


def main():
    # Creating and setuping environment
    env = turtlesim_env_multi.provide_env()
    env.setup(ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)
    env.DETECT_COLLISION = DETECT_COLLISIONS

    # Setuping agents
    agents = env.reset()
    ag_name = list(agents.keys())

    # Setuping dqn and loading model
    dqn = DqnMulti(env, 'simulate_multi')
    dqn.model = load_model(MODEL_FILEPATH)

    # Setuping data for steps
    current_states = {tname: agent.map for tname, agent in env.agents.items()}
    last_states = {tname: agent.map for tname, agent in env.agents.items()}
    agent_episode = {tname: i for i, tname in enumerate(env.agents)}

    # Running steps
    controls = {}
    for move in range(MOVES):
        for tname in agents:
            controls[tname] = np.argmax(dqn.decision_multi(dqn.model, last_states[tname], current_states[tname]))

        actions = {tname: dqn.ctl2act(control) for tname, control in controls.items()}
        scene = env.step(actions)

        for tname, (new_state, reward, done) in scene.items():
            last_states[tname] = current_states[tname]
            current_states[tname] = new_state


if __name__ == "__main__":
    main()
