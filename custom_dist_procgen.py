import numpy as np
from procgen.env import ProcgenEnv

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style
# mpl.use('Agg')
# mpl.use("macOSX")
mpl.style.use('seaborn')


class CustomDistProcgen(object):
    def __init__(self,
                 game_list=['coinrun'], 
                 num_envs=1, 
                 num_levels=0, 
                 distribution_mode='easy', 
                 start_level=0, 
                 paint_vel_info=False, 
                 env_dist=None):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        assert len(game_list) > 0, 'Arg game_list cannot be an empty list.'

        self.num_envs = num_envs
        self.game_list = game_list

        num_games = len(game_list)
        
        if env_dist is None:
            env_dist = np.ones(num_games)/num_games
        else:
            assert len(env_dist) == num_games, \
                'Args env_dist and game_list must be the same length.'
        self.env_dist = None
        self.update_env_dist(env_dist)
        self.env_name_sample = None

        self.venv = ProcgenEnv(
            env_name=game_list[0],
            num_envs=num_envs, 
            num_levels=num_levels,
            distribution_mode=distribution_mode,
            start_level=start_level,
            paint_vel_info=paint_vel_info,
        )

        self.reset()

    def update_env_dist(self, env_dist):
        if self.env_dist is not None:
            assert len(env_dist) == len(self.env_dist), \
                'New env_dist must match length of existing env_dist.'

        if not isinstance(env_dist, np.ndarray):
            env_dist = np.array(env_dist)

        env_dist = env_dist/env_dist.sum()

        self.env_dist = env_dist

    def _resample_env_names(self, n=None):
        if n == None:
            n = self.num_envs

        env_names = np.random.choice(self.game_list, 
            size=n, replace=True, p=self.env_dist)

        return env_names

    def reset(self):
        env_names = self._resample_env_names()
        self.env_name_sample = env_names

        for i, name in enumerate(env_names):
            self.venv.reset_at_index(i, env_name=name)

        return self.venv.reset()

    def step(self, actions):
        obs, reward, done, info = self.venv.step(actions)

        # Resample the env per index based on current env dist when that env is done.
        # Note that in this case, info['level_seed'] will return the level_seed for
        # the level on whose state the step was taken.
        for i, done_ in enumerate(done):
            info[i]['env_name'] = self.env_name_sample[i]
            if done:
                next_env_name = self._resample_env_names(n=1)[0]
                
                self.env_name_sample[i] = next_env_name
                info[i]['next_env_name'] = next_env_name

                obs_ = self.venv.reset_at_index(i, env_name=next_env_name)
                for i, (k,v) in enumerate(obs.items()):
                    obs[k][i] = obs_[k][i]

        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.venv, name)


if __name__ == "__main__":
    num_envs = 2

    venv = CustomDistProcgen(
            num_envs=num_envs,
            game_list=['coinrun', 'bigfish', 'ninja', 'chaser'],    
            num_levels=1, 
            # start_level=np.random.randint(10000),
            start_level=0,
            distribution_mode='hard',
            paint_vel_info=False,
            env_dist=[0.5,]*4)

    obs = venv.reset()    

    num_samples = 10
    f, axarr = plt.subplots(2,num_samples) 

    for i in range(num_samples):
        obs = venv.reset()
        axarr[0][i].imshow(obs['rgb'][0])

    venv.update_env_dist([1, 0, 0, 0])

    for i in range(num_samples):
        obs = venv.reset()
        axarr[1][i].imshow(obs['rgb'][0])

    plt.show()
