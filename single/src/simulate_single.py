import numpy as np
import tensorflow as tf

import turtlesim_env_single
from dqn_single import DqnSingle

MOVES = 1000
NUMBER_OF_AGENTS = 1
ROUTES_FILENAME = '/root/siu/routes_single.csv'
MODEL_FILEPATH = '/root/siu/models/model_single.h5'


def main():
    env = turtlesim_env_single.provide_env()
    env.setup(routes_fname=ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)

    agents = env.reset()
    tname = list(agents.keys())[0]

    dqn = DqnSingle(env, 'simulate_single')
    dqn.model = tf.keras.models.load_model(MODEL_FILEPATH)

    current_state = agents[tname].map
    last_state = [i.copy() for i in current_state]
    for step in range(MOVES):
        control = np.argmax(dqn.decision_single(dqn.model, last_state, current_state))
        last_state = current_state
        current_state, reward, done = env.step({tname: dqn.ctl2act(int(control))})


if __name__ == "__main__":
    main()
