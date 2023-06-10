import tensorflow as tf

import turtlesim_env_single
from dqn_single import DqnSingle

NUMBER_OF_AGENTS = 1
ROUTES_FILENAME = '/root/siu/routes_single.csv'

LOAD_MODEL = False
MODEL_FILEPATH = '/root/siu/models/model_single.h5'


def main():
    env = turtlesim_env_single.provide_env()
    env.setup(routes_fname=ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)

    agents = env.reset()
    tnames = list(agents.keys())
    dqn = DqnSingle(env, 'train_single')
    if LOAD_MODEL:
        dqn.model = tf.keras.models.load_model(MODEL_FILEPATH)
    else:
        dqn.make_model_single()

    for tname in tnames:
        dqn.train_main_single(tname, save_model=True)


if __name__ == "__main__":
    main()
