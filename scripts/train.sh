#/bin/bash

# use a loop to train all the models
echo "-----Training-----"
configs=(full_w_her)
targets=(bca lcca)
phantom=phantom4
n_runs=2
n_timesteps=600000
for run in $(seq 1 $n_runs); do
    for config in ${configs[@]}; do
        for target in ${targets[@]}; do
            echo "Training $config on $phantom - $target"
            python rl/sb3/train.py --config $config --target $target  --phantom $phantom --n-runs $n_runs --n-timesteps $n_timesteps
        done
    done
done

