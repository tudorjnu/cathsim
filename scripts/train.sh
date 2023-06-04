#/bin/bash

# use a loop to train all the models
echo "-----Training-----"
configs=(full full_w_her internal pixels internal_pixels pixels_mask)
targets=(lcca)
phantom=phantom3
n_runs=2
n_timesteps=600000
for config in ${configs[@]}; do
    for target in ${targets[@]}; do
        echo "Training $config on $phantom - $target - run $run"
        python rl/sb3/train.py --config $config --target $target  --phantom $phantom --n-runs $n_runs --n-timesteps $n_timesteps
    done
done

