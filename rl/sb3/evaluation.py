from pathlib import Path
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


def calculate_total_distance(positions):
    return np.sum(distance(positions[1:], positions[:-1]))


def analyze_model(result_path: Path, optimal_path_length: float = 15.73) -> OrderedDict:
    data = np.load(result_path, allow_pickle=True)
    episodes = data['results']
    if len(episodes) == 0:
        return None

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


def aggregate_results(eval_path: Path = None, output_path: Path = None) -> pd.DataFrame:
    eval_path = eval_path or EVALUATION_PATH
    output_path = output_path or EVALUATION_PATH / 'results.csv'

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


def collate_results(experiment_path: Path = None, evaluation_path: Path = None) -> None:
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
                        print(f'Collating results for {phantom.name}/{target.name}/{experiment.name}')
                        path = Path(f'{phantom.name}/{target.name}')
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
                if not algorithm.stem == 'bc':
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
        print(algorithm_path)
        if (eval_path / (model_name + '.npz')).exists():
            continue
        print(f'Evaluating {model_name} in {algorithm_path} for {n_episodes} episodes.')
        model = SAC.load(model_filename)
        config = get_config(algorithm_path.stem)
        env = make_env(config)
        evaluation_data = evaluate_policy(model, env, n_episodes=n_episodes)
        np.savez_compressed(eval_path / f'{model_name}.npz', **evaluation_data)


def parse_tensorboard_log(path: Path):
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


def get_tensorboard_logs(path: Path):
    for experiment in EXPERIMENT_PATH.iterdir():
        if not experiment.is_dir():
            continue
        tensorboard_path = experiment / 'logs'
        if not tensorboard_path.exists():
            continue
        for model in tensorboard_path.iterdir():
            if not model.is_dir():
                continue
            for log in model.iterdir():
                if log.suffix != '.tfevents':
                    continue
                yield parse_tensorboard_log(log)


def get_experiment_tensorboard_logs(experiment_name: str):
    _, experiment_log_path, _ = make_experiment(experiment_name)
    logs = []
    for model in experiment_log_path.iterdir():
        if not model.is_dir():
            continue
        for log in model.iterdir():
            tensorboard_log = parse_tensorboard_log(log.as_posix())
            if tensorboard_log is not None:
                logs.append(tensorboard_log)
    return logs


def collate_experiment_tensorboard_logs(experiment_name: str, n_interpolations: int = 30):
    logs = get_experiment_tensorboard_logs(experiment_name)
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


def plot_error_line_graph(mean, stdev, label, color='C0'):
    x = mean.index
    plt.plot(x, mean, color=color, label=label)
    plt.fill_between(x, mean - stdev, mean + stdev, alpha=0.3, color=color)


if __name__ == '__main__':

    # evaluate_model(EXPERIMENT_PATH / 'low_tort' / 'bca' / 'full', n_episodes=10)
    # evaluate_models()
    # lcca_evaluation.mkdir(exist_ok=True)
    collate_results()
    dataframe = aggregate_results()
    dataframe.to_csv(EVALUATION_PATH / 'results_2.csv', index=False)
    # make column names title case, without underscores
    dataframe.columns = [column.replace('_', ' ').title() for column in dataframe.columns]
    print(dataframe.columns)

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
                   'Episode Length (s)', 'Safety \%', 'Success \%', 'SPL \%',]
    dataframe['Force (N)'] = '$' + dataframe['Force'].astype(str) + ' \\pm ' + dataframe['Force Std'].astype(str) + '$'
    dataframe['Path Length (mm)'] = '$' + dataframe['Path Length'].astype(str) + ' \\pm ' + dataframe['Path Length Std'].astype(str) + '$'
    dataframe['Episode Length (s)'] = '$' + dataframe['Episode Length'].astype(str) + ' \\pm ' + dataframe['Episode Length Std'].astype(str) + '$'
    dataframe['Safety \%'] = '$' + dataframe['Safety'].astype(str) + ' \\pm ' + dataframe['Safety Std'].astype(str) + '$'
    dataframe['Success \%'] = '$' + dataframe['Success'].astype(str) + ' \\pm ' + dataframe['Success Std'].astype(str) + '$'
    dataframe['SPL \%'] = '$' + dataframe['Spl'].astype(str) + '$'
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
