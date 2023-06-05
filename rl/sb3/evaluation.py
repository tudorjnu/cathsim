from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
from tqdm import tqdm
from typing import OrderedDict

from cathsim.cathsim.env_utils import distance
from rl.sb3.sb3_utils import make_experiment
from rl.sb3.sb3_utils import EXPERIMENT_PATH, EVALUATION_PATH

from stable_baselines3.common.base_class import BaseAlgorithm

from typing import Union, Callable, Optional, Any, Dict, List, Tuple


def calculate_total_distance(positions):
    return np.sum(distance(positions[1:], positions[:-1]))


def load_human_trajectories(path: Path, flatten=False, mapping: dict = None):
    trajectories = {}
    for episode in path.iterdir():
        trajectory_path = episode / 'trajectory.npz'
        if not trajectory_path.exists():
            continue
        episode_data = np.load(episode / 'trajectory.npz', allow_pickle=True)
        episode_data = dict(episode_data)
        if flatten:
            for key, value in episode_data.items():
                if mapping is not None:
                    if key in mapping:
                        key = mapping[key]
                if key == 'time':
                    continue
                trajectories.setdefault(key, []).extend(value)
        else:
            if mapping is not None:
                for key, value in mapping.items():
                    episode_data[mapping[key]] = episode_data.pop(key)
            trajectories[episode.name] = episode_data
    if flatten:
        for key, value in trajectories.items():
            trajectories[key] = np.array(value)

    return trajectories


def human_data_loader(path: Path) -> list:
    trajectories = load_human_trajectories(path, flatten=False, mapping={'force': 'forces'})
    return list(trajectories.values())


def analyze_model(result_path: Path, optimal_path_length: float = 15.73,
                  human: bool = False, human_data_fn: callable = None) -> OrderedDict:
    if not human:
        data = np.load(result_path, allow_pickle=True)
        if 'results' not in data:
            episodes = data
        else:
            episodes = data['results']
        if len(episodes) == 0:
            return None
    else:
        episodes = human_data_fn(result_path)

    algo_results = []
    for episode in episodes:
        episode_forces = episode['forces']
        episode_head_positions = episode['head_positions']
        episode_length = len(episode_head_positions)
        total_distance = calculate_total_distance(episode_head_positions)

        algo_results.append([
            episode_forces.mean(),
            total_distance * 100,
            episode_length,
            1 - np.sum(np.where(episode_forces > 2, 1, 0)) / episode_length,
            get_curvature(episode_head_positions),
            np.sum(np.where(episode_length <= 300, 1, 0)),
        ])

    algo_results = np.array(algo_results)
    mean, std = algo_results.mean(axis=0), algo_results.std(axis=0)
    spl = optimal_path_length / np.maximum(algo_results[:, 1], optimal_path_length)
    mean_spl = np.mean(spl).round(2)

    summary_results = OrderedDict(
        force=mean[0].round(2),
        force_std=std[0].round(2),
        path_length=mean[1].round(2),
        path_length_std=std[1].round(2),
        episode_length=mean[2].round(2),
        episode_length_std=std[2].round(2),
        safety=mean[3].round(2),
        safety_std=std[3].round(2),
        curv=mean[4].round(2),
        curv_std=std[4].round(2),
        success=mean[5].round(2),
        success_std=std[5].round(2),
        spl=mean_spl,
    )
    return summary_results


def aggregate_results(eval_path: Path = None, output_path: Path = None,
                      verbose: bool = False) -> pd.DataFrame:
    eval_path = eval_path or EVALUATION_PATH

    if verbose:
        print(f'Analyzing {"experiment".ljust(30)} {"phantom".ljust(30)} {"target".ljust(30)}')

    dataframe = pd.DataFrame()

    phantoms = [p for p in eval_path.iterdir() if p.is_dir()]
    for phantom in phantoms:
        targets = [t for t in phantom.iterdir() if t.is_dir()]
        for target in targets:
            files = [f for f in target.iterdir() if f.suffix == '.npz']
            for file in files:
                print(f'Analyzing {file.stem.ljust(30)} {phantom.stem.ljust(30)} {target.stem.ljust(30)}')
                results = analyze_model(file)
                if results is not None:
                    if dataframe.empty:
                        dataframe = pd.DataFrame(columns=['phantom', 'target', 'algorithm', *results.keys()])
                    results_dataframe = pd.DataFrame({
                        'phantom': phantom.stem,
                        'target': target.stem,
                        'algorithm': file.stem,
                        **results
                    }, index=[0])
                    dataframe = pd.concat([dataframe, results_dataframe], ignore_index=True)
    return dataframe


def get_curvature(points: np.ndarray) -> np.ndarray:
    # Calculate the first and second derivatives of the points
    first_deriv = np.gradient(points)
    second_deriv = np.gradient(first_deriv)

    # Calculate the norm of the first derivative
    norm_first_deriv = np.linalg.norm(first_deriv, axis=0)

    # Calculate the curvature
    curvature = np.linalg.norm(np.cross(first_deriv, second_deriv), axis=0) / np.power(norm_first_deriv, 3)

    # Return the curvature
    return curvature.mean()


# TODO: refactor
def plot_path(filename):
    def point2pixel(point, camera_matrix: np.ndarray = None):
        """Transforms from world coordinates to pixel coordinates for a
        480 by 480 image"""
        camera_matrix = np.array([
            [-5.79411255e+02, 0.00000000e+00, 2.39500000e+02, - 5.33073376e+01],
            [0.00000000e+00, 5.79411255e+02, 2.39500000e+02, - 1.08351407e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, - 1.50000000e-01]
        ])
        x, y, z = point
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

        return np.array([round(xs / s), round(ys / s)], np.int8)

    data = np.load(EVALUATION_PATH / filename, allow_pickle=True)
    data = {key: value.item() for key, value in data.items()}
    paths = {}
    for episode, values in data.items():
        episode_head_positions = np.apply_along_axis(point2pixel, 1, values['head_positions'])
        paths[episode] = episode_head_positions
        break

    import matplotlib.pyplot as plt
    import cv2
    curv = get_curvature(paths['0'])
    # drop nan values
    curv = curv[~np.isnan(curv)]
    mean_curv = np.round(np.mean(curv), 2)
    std_curv = np.round(np.std(curv), 2)

    print(mean_curv, std_curv)
    exit()
    image = cv2.imread('./figures/phantom.png', 0)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    for episode, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], label=f'Episode {episode}')
    # image_size = 80
    ax.set_ylim(480, None)
    # ax.legend()
    ax.axis('off')
    plt.show()


def collate_human_trajectories(path: Path, mapping: dict = None, save_path: Path = None):
    """ Collate the human trajectories into a single numpy array.

    path: (Path) The path to the human trajectories.
    mapping: (dict) A mapping from the keys in the trajectory to the keys in the collated array.
    """
    mapping = mapping or {'force': 'forces'}
    trajectories = {}
    for i, episode in enumerate(path.iterdir()):
        trajectory_path = episode / 'trajectory.npz'
        if not trajectory_path.exists():
            continue
        episode_data = np.load(episode / 'trajectory.npz', allow_pickle=True)
        episode_data = dict(episode_data)
        if mapping is not None:
            for key, value in mapping.items():
                episode_data[mapping[key]] = episode_data.pop(key)
        trajectories[str(i)] = episode_data
    trajectories = list(trajectories.values())
    if save_path is not None:
        save_path = EVALUATION_PATH / save_path
        np.savez(save_path, results=trajectories)
    return trajectories


def collate_experiment_results(experiment_path: Path) -> list:
    """
    Collate the results of all the runs of an experiment into a single numpy array.

    :param experiment_name: (str) The name of the experiment.
    :return: (np.ndarray) The collated results.
    """
    experiment_path = EXPERIMENT_PATH / experiment_path

    _, _, eval_path = make_experiment(experiment_path)
    collated_results = []
    for seed_evaluation in eval_path.iterdir():
        if seed_evaluation.suffix != '.npz':
            continue
        else:
            evaluation_results = np.load(seed_evaluation, allow_pickle=True)
            evaluation_results = [value.item() for value in evaluation_results.values()]
            collated_results.extend(evaluation_results)
    return collated_results


def collate_results(experiment_path: Path = None, evaluation_path: Path = None,
                    verbose: bool = False) -> None:
    """
    Check the results of all the seeds of all the experiments and collate them into a single numpy array.

    :param experiment_path: (Path) The path to the experiments.
    :param evaluation_path: (Path) The path to the evaluation results.
    """
    if experiment_path is None:
        experiment_path = EXPERIMENT_PATH
    if evaluation_path is None:
        evaluation_path = EVALUATION_PATH
    for phantom in experiment_path.iterdir():
        if phantom.is_dir() is False:
            continue
        for target in phantom.iterdir():
            if target.is_dir() is False:
                continue
            else:
                for experiment in target.iterdir():
                    if experiment.is_dir() is False:
                        continue
                    else:
                        if verbose:
                            print(f'Collating results for {phantom.name}/{target.name}/{experiment.name}')
                        path = Path(f'{phantom.name}/{target.name}')
                        if experiment.name == 'human':
                            experiment_results = collate_human_trajectories(path / experiment.name)
                        else:
                            experiment_results = collate_experiment_results(path / experiment.name)
                        (EVALUATION_PATH / path).mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(EVALUATION_PATH / path / f'{experiment.name}.npz', results=experiment_results)


def evaluate_policy(model: BaseAlgorithm, env: gym.Env, n_episodes: int = 10) -> dict:
    """
    Evaluate the performance of a policy.

    :param model  type_aliases.PolicyPredictor: The policy to evaluate.
    :param env gym.Env: The environment to evaluate the policy in.
    :param n_episodes int: The number of episodes to evaluate the policy for.
    :return: dict: The evaluation data.

    Example:
    >>> from stable_baselines3 import SAC
    >>> from cathsim.cathsim.env_utils import make_env, get_config
    >>>
    >>> algorithm = 'full'
    >>> config = get_config(algorithm)
    >>> env = make_env(config)
    >>> model = SAC.load(f'./models/{algorithm}/best_model.zip')
    >>> evaluation_data = evaluate_policy(model, env)
    """
    evaluation_data = {}
    for episode in tqdm(range(n_episodes)):
        observation = env.reset()
        done = False
        head_positions = []
        forces = []
        head_pos = env.head_pos.copy()
        head_positions.append(head_pos)
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            head_pos_ = info['head_pos']
            forces.append(info['forces'])
            head_positions.append(head_pos_)
            head_pos = head_pos_
        evaluation_data[str(episode)] = dict(
            forces=np.array(forces),
            head_positions=np.array(head_positions),
        )
    return evaluation_data


def evaluate_models(experiments_path: Path = None, n_episodes=10,
                    phantom_name: str = None, target_name: str = None, algorithm_name: str = None):
    """
    Evaluate the performance of all the models in the experiments directory.

    :param experiments_path Path: The path to the experiments directory.
    :param n_episodes int: The number of episodes to evaluate the policy for.
    """
    if not experiments_path:
        experiments_path = EXPERIMENT_PATH

    phantoms = [phantom for phantom in experiments_path.iterdir() if
                (phantom.is_dir() and (phantom_name is None or phantom.name == phantom_name))]
    for phantom in phantoms:
        targets = [target for target in phantom.iterdir() if
                   (target.is_dir() and (target_name is None or target.name == target_name))]
        for target in targets:
            algorithms = [algorithm for algorithm in target.iterdir() if
                          (algorithm_name is None or algorithm.name == algorithm_name)]
            for algorithm in algorithms:
                evaluate_model(algorithm, n_episodes)


def evaluate_model(algorithm_path, n_episodes=10):
    """
    Evaluate the performance of a model.

    :param model_name str: The name of the model to evaluate.
    :param n_episodes int: The number of episodes to evaluate the policy for.
    """
    from stable_baselines3 import SAC
    from rl.sb3.sb3_utils import get_config, make_experiment
    from cathsim.cathsim.env_utils import make_env

    model_path, _, eval_path = make_experiment(algorithm_path)

    for model_filename in model_path.iterdir():
        model_name = model_filename.stem
        if (eval_path / (model_name + '.npz')).exists():
            continue
        print(f'Evaluating {model_name} in {algorithm_path} for {n_episodes} episodes.')
        config = get_config(algorithm_path.stem)
        config['task_kwargs']['phantom'] = algorithm_path.parent.parent.stem
        config['task_kwargs']['target'] = algorithm_path.parent.stem
        if algorithm_path.stem == 'bc':
            config['wrapper_kwargs']['channel_first'] = True
        env = make_env(config)
        if algorithm_path.stem == 'bc':
            from scratch.bc.custom_networks import CnnPolicy
            from cathsim.wrappers import Dict2Array
            import torch as th
            env = Dict2Array(env)

            model = CnnPolicy(observation_space=env.observation_space,
                              action_space=env.action_space,
                              lr_schedule=lambda _: th.finfo(th.float32).max,
                              ).load(model_path / 'bc')
        else:
            model = SAC.load(model_filename)
        evaluation_data = evaluate_policy(model, env, n_episodes=n_episodes)
        np.savez_compressed(eval_path / f'{model_name}.npz', **evaluation_data)


def parse_tensorboard_log(path: Path):
    """ Parse a tensorboard log.

    :param path Path: The path to the tensorboard log.
    :return pd.DataFrame: The parsed tensorboard log.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd

    tag = 'rollout/ep_len_mean'
    acc = EventAccumulator(path)
    acc.Reload()
    # check if the tag exists
    if tag not in acc.Tags()['scalars']:
        print("Tag not found")
        return
    df = pd.DataFrame(acc.Scalars(tag))
    df = df.drop(columns=['wall_time']).groupby('step').mean()
    return df


def get_experiment_tensorboard_logs(path: Path, n_interpolations: int = None) -> Tuple[List[pd.DataFrame], Tuple]:
    """ Get the tensorboard logs for an experiment.

    :param path Path: The path to the experiment.
    :param n_interpolations int: The number of interpolations to use.
    :return List[pd.DataFrame]: A list of dataframes containing the tensorboard logs.
    """

    _, log_path, _ = make_experiment(path)
    logs = []
    log_paths = [log for log in log_path.iterdir() if log.is_dir()]
    for log in log_paths:
        tensorboard_log = parse_tensorboard_log(log.as_posix())
        assert len(tensorboard_log) != 0, f'Log {log} is empty.'
        if len(tensorboard_log) < 150:
            print(f'Log {log.parent.parent.stem}-{log.stem} is too short({len(tensorboard_log)} < 500).')
            continue
        logs.append(tensorboard_log)

    if n_interpolations:
        try:
            logs = [log.reindex(np.arange(0, 600_000, 600_000 / n_interpolations), method='nearest') for log in logs]
            if len(logs) == 0:
                mean = None
                stdev = None
            else:
                mean = pd.concat(logs, axis=0).groupby(level=0).mean().squeeze()
                stdev = pd.concat(logs, axis=0).groupby(level=0).std().squeeze()
        except Exception as e:
            print(e)
            mean = None
            stdev = None
    return logs, (mean, stdev)


def collate_experiment_tensorboard_logs(path: str, n_interpolations: int = 30):
    logs = get_experiment_tensorboard_logs(path)
    logs = [log.reindex(np.arange(0, 600_000, 600_000 / n_interpolations), method='nearest') for log in logs]
    print(len(logs))
    mean = pd.concat(logs, axis=0).groupby(level=0).mean().squeeze()
    stdev = pd.concat(logs, axis=0).groupby(level=0).std().squeeze()
    return mean, stdev


def collate_experiments_tensorboard_logs(experiments_path: Path = None, n_interpolations: int = 30):
    if not experiments_path:
        experiments_path = EXPERIMENT_PATH
    results = {}
    for experiment in experiments_path.iterdir():
        if not experiment.is_dir():
            continue
        mean, stdev = collate_experiment_tensorboard_logs(experiment.name, n_interpolations)
        results[experiment.name] = dict(mean=mean, stdev=stdev)
    return results


def plot_error_line_graph(ax: plt.Axes, mean: pd.Series, std: pd.Series,
                          color: str = 'C0', label: str = None, **kwargs):
    """ Plot a line graph with error bars.

    :param ax plt.Axes: The axes to plot on.
    :param mean pd.Series: The mean values.
    :param std pd.Series: The standard deviation values.
    :param color str: The color of the line.
    :param label str: The label of the line.
    :return plt.Axes: The axes.
    """

    x = mean.index
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=color, **kwargs)


def plot_human_line(ax, phantom, target, n_interpolations=30):
    """ Plot the human line.

    :param ax plt.Axes: The axes to plot on.
    :param phantom str: The phantom to plot.
    :param target str: The target to plot.
    :param n_interpolations int: The number of interpolations to use.
    :return plt.Axes: The axes.
    """
    experiment_data = pd.read_csv(EVALUATION_PATH / 'results_2.csv')
    human_data = experiment_data[(experiment_data['algorithm'] == 'human') & (experiment_data['phantom'] == phantom) & (experiment_data['target'] == target)]
    mean = human_data['episode_length'].to_numpy()[0]
    y_std = human_data['episode_length_std'].to_numpy()[0]

    x = np.arange(0, 600_000, 600_000 / n_interpolations)
    y = np.full(n_interpolations, mean)
    ax.plot(x, y + y_std, linewidth=0.5, color='C0')
    ax.plot(x, y - y_std, linewidth=0.5, color='C0')

    ax.plot(x, y, label='Human', linestyle='--', color='C0', linewidth=1)
    ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)


if __name__ == '__main__':

    # evaluate_model(EXPERIMENT_PATH / 'low_tort' / 'bca' / 'full', n_episodes=10)
    evaluate_models()
    # lcca_evaluation.mkdir(exist_ok=True)

    collate_results(verbose=True)
    dataframe = aggregate_results()
    print(dataframe)
    dataframe.to_csv(EVALUATION_PATH / 'results_2.csv', index=False)
    exit()
    # make column names title case, without underscores
    dataframe.columns = [column.replace('_', ' ').title() for column in dataframe.columns]

    columns = ['Phantom', 'Target', 'Algorithm', 'Force', 'Force Std', 'Path Length',
               'Path Length Std', 'Episode Length', 'Episode Length Std', 'Safety',
               'Safety Std', 'Curv', 'Curv Std', 'Success', 'Success Std', 'Spl']
    # remove curv and curv std
    columns.pop(11)
    columns.pop(12)
    # make sure all the numbers are formatted with two 00s after the decimal point
    # using :.2f
    dataframe = dataframe.applymap(lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    new_columns = ['Phantom', 'Target', 'Algorithm', 'Force (N)', 'Path Length (mm)',
                   'Episode Length (s)', 'Safety %', 'Success %', 'SPL %',]
    dataframe['Force (N)'] = '$' + dataframe['Force'].astype(str) + ' \\pm ' + dataframe['Force Std'].astype(str) + '$'
    dataframe['Path Length (mm)'] = '$' + dataframe['Path Length'].astype(str) + ' \\pm ' + dataframe['Path Length Std'].astype(str) + '$'
    dataframe['Episode Length (s)'] = '$' + dataframe['Episode Length'].astype(str) + ' \\pm ' + dataframe['Episode Length Std'].astype(str) + '$'
    dataframe['Safety %'] = '$' + dataframe['Safety'].astype(str) + ' \\pm ' + dataframe['Safety Std'].astype(str) + '$'
    dataframe['Success %'] = '$' + dataframe['Success'].astype(str) + ' \\pm ' + dataframe['Success Std'].astype(str) + '$'
    dataframe['SPL %'] = '$' + dataframe['Spl'].astype(str) + '$'
    # format the elements of the columns
    # drop the row where the phantom is low_tort
    dataframe = dataframe[dataframe['Phantom'] != 'low_tort']
    formatters = {
        'Phantom': lambda x: 'Type-I Aortic Arch' if x == 'phantom3' else 'Type-II Aortic Arch',
        'Target': lambda x: x.upper(),
        'Algorithm': lambda x: x.replace('_', ' ').title(),
    }
    # make the targets, which are bca and lcca, to be as a second collumn

    for column in new_columns:
        if column in formatters:
            dataframe[column] = dataframe[column].apply(formatters[column])

    # multiindex based on the phantom and target
    dataframe = dataframe.set_index(['Phantom', 'Target', 'Algorithm'])

    print(dataframe)
    print(dataframe.to_latex(float_format="%.2f", sparsify=True,
                             columns=new_columns, column_format='cccrrrrrr', escape=False,
                             formatters={
                                 'Phantom': lambda x: x.upper(),
                                 'Target': lambda x: x.upper(),
                             }))
