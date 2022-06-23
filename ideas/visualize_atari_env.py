from absl import app
from absl import flags
from absl import logging

import matplotlib.pyplot as plt

from deep_rl_zoo import gym_env

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')


def main(argv):

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            seed=1,
            noop_max=30,
            terminal_on_life_loss=True,
            # scale_obs=False,
            clip_reward=True,
        )

    env = environment_builder()

    obs = env.reset()

    for _ in range(50):
        a_t = env.action_space.sample()

        obs, _, done, _ = env.step(a_t)

        if done:
            break

    plt.title("Game image")
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()

    plt.title("Agent observation (4 frames left to right)")
    obs = obs.transpose(1, 2, 0)  # switch to channel last
    plt.imshow(obs.transpose([0, 2, 1]).reshape([obs.shape[0], -1]))
    plt.show()


if __name__ == '__main__':
    app.run(main)
