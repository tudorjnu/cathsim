from setuptools import setup, find_packages

extra_dev = [
    'opencv-python',
    'matplotlib',
]


setup(
    name='cathsim',
    version='dev',
    url='git@github.com:tudorjnu/packaging_test.git',
    author='Author',
    author_email='my_email',
    packages=find_packages(
        exclude=[
            'rl'
        ]
    ),
    install_requires=[
        'dm_control',
        'gym==0.21.*',
    ],
    extras_require={
        'dev': extra_dev,
    },
    entry_points={
        'console_scripts': [
            'run_env=cathsim.env:run_env',
        ],
    },
)
