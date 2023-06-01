#/bin/bash

# use a loop to train all the models
echo "-----Training-----"
configs=(full internal internal_pixels pixels pixels_mask full_w_her full_w_her_w_sampling)
targets=(bca)
phantom=phantom4
n_runs=2
n_timesteps=1000000
for config in ${configs[@]}; do
    for target in ${targets[@]}; do
        echo "Training $config on $phantom - $target"
        python rl/sb3/train.py --config $config --target $target  --phantom $phantom --n-runs $n_runs --n-timesteps $n_timesteps
    done
done

