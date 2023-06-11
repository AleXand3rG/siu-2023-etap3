import tensorflow as tf

import turtlesim_env_multi
from dqn_multi import DqnMulti

NUMBER_OF_AGENTS = 8
NUMBER_OF_EPISODES = 4000  # pierwsze 2000 epizodów bez kolizji, kolejne 2000 epizodów z kolizjami
ROUTES_FILENAME = '/root/siu/routes_multi.csv'

LOAD_MODEL = False
LOAD_MODEL_FILEPATH = '/root/siu/models/model_multi.h5'

new_model_filepath: str


def setup_dqnm(env) -> DqnMulti:
    global new_model_filepath
    dqnm = DqnMulti(env, 'train_multi')
    dqnm.EPISODES_MAX = NUMBER_OF_EPISODES
    dqnm.REPLAY_MEM_SIZE_MAX = 2 * NUMBER_OF_EPISODES
    dqnm.REPLAY_MEM_SIZE_MIN = NUMBER_OF_EPISODES / 2
    new_model_filepath = f"/root/siu/models/{dqnm.xid()}.h5"
    dqnm.model_filepath = new_model_filepath
    if LOAD_MODEL:
        dqnm.model = tf.keras.models.load_model(LOAD_MODEL_FILEPATH)
    else:
        dqnm.make_model_multi()
    return dqnm


def main():
    global new_model_filepath
    # === Set the environment up === #
    print("Preparing first stage of training...")
    env = turtlesim_env_multi.provide_env()
    env.setup(routes_fname=ROUTES_FILENAME, agent_cnt=NUMBER_OF_AGENTS)
    env.reset()
    dqnm = setup_dqnm(env)

    # === First 2000 episodes === #
    print(f"First stage of training ({NUMBER_OF_EPISODES / 2} episodes) started...")
    dqnm.train_main_multi(save_model=True)

    # === Turn on collisions and run next episodes === #
    print(f"First {NUMBER_OF_EPISODES / 2} episodes finished. ")
    print(f"Turning on collisions and starting next stage of training...")
    env.reset()
    env.DETECT_COLLISION = True
    dqnm2 = setup_dqnm(env)
    dqnm2.model_filepath = new_model_filepath
    dqnm2.load_trained_model()

    # === Second 2000 episodes === #
    print(f"Second stage of training ({NUMBER_OF_EPISODES / 2} episodes) started...")
    dqnm2.train_main_multi(save_model=True)


if __name__ == "__main__":
    main()
