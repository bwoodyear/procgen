import gym
import numpy as np


class Scalarize:
    """
    Convert a VecEnv into an Env, this is basically the opposite of DummyVecEnv

    There is a minor difference between this and a normal Env, which is that
    the final observation (when done=True) does not exist, and instead you will
    receive the second to last observation a second time due to how the VecEnv
    interface handles resets.  In addition, you are cannot step this
    environment after done is True, since that is not possible for VecEnvs.
    """

    def __init__(self, venv, env_idx=0) -> None:
        assert venv.num_envs > env_idx
        self.env_idx = env_idx
        self._venv = venv
        self._waiting_for_reset = True
        self._previous_obs = None
        self._next_obs = None
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space
        self.metadata = self._venv.metadata
        if hasattr(self._venv, "spec"):
            self.spec = self._venv.spec
        else:
            self.spec = None
        if hasattr(self._venv, "reward_range"):
            self.reward_range = self._venv.reward_range
        else:
            self.reward_range = None

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            # dict space
            scalar_obs = {}
            for k, v in obs.items():
                scalar_obs[k] = v[self.env_idx]
            return scalar_obs
        else:
            return obs[self.env_idx]

    def reset(self):
        if self._waiting_for_reset and self._next_obs is not None:
            # if we just reset, we already have the next obs that should be returned by reset, so return it directly
            # this is an optimization for procgen which doesn't support reset()
            obs = self._next_obs
        else:
            # self._venv.seed(40, self.env_idx)
            obs = self._venv.reset()
            self._previous_obs = obs
        self._waiting_for_reset = False
        return self._process_obs(obs)

    def step(self, action):
        assert not self._waiting_for_reset
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = np.array([action], dtype=self._venv.action_space.dtype)
        else:
            action = np.expand_dims(action, axis=0)
        obs, rews, dones, infos = self._venv.step(action)
        if dones[self.env_idx]:
            # self._venv.seed(40, self.env_idx)
            # self._waiting_for_reset = True
            self._next_obs = obs
            obs = self._previous_obs
        else:
            self._previous_obs = obs
        return self._process_obs(obs), rews[self.env_idx], dones[self.env_idx], infos[self.env_idx]

    def render(self, mode="human"):
        if mode == "human":
            return self._venv.render(mode=mode)
        else:
            assert mode == "rgb_array"
            return self._venv.get_images()[self.env_idx]

    def close(self):
        return self._venv.close()

    def seed(self, seed=None, env_idx=-1):
        return self._venv.seed(seed, env_idx)

    @property
    def unwrapped(self):
        return self

    def __repr__(self):
        return f"<Scalarize venv={self._venv}>"
