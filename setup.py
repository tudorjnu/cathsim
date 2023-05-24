from setuptools import setup, find_packages

extra_dev = [
    'opencv-python',
    'matplotlib',
]

extra_rl = [
    'torch',
    'stable-baselines3==1.8.0',
    'sb3_contrib',
    'imitation',
    'tqdm',
    'rich',
    'mergedeep',
    'progressbar2',
]

extra = extra_dev + extra_rl


setup(
    name='cathsim',
    version='dev',
    url='git@github.com:tudorjnu/packaging_test.git',
    author='Tudor Jianu',
    author_email='tudorjnu@gmail.com',
    packages=find_packages(
        exclude=[
            'tests',
            'scripts',
            'notebooks',
            'figures',
        ]
    ),
    # setuptools==58.2.0
    install_requires=[
        'gym==0.21.*',
        'dm-control',
        'pyyaml',
    ],
    extras_require={
        'dev': extra_dev,
        'rl': extra_rl,
        'all': extra,
    },
    entry_points={
        'console_scripts': [
            'run_env=cathsim.cathsim.env:run_env',
            'record_traj=rl.expert.utils:cmd_record_traj',
            'visualize_agent=rl.sb3.sb3_utils:cmd_visualize_agent',
        ],
    },
)
