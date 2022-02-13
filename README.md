# Installation Procedure

1. Download [MuJoCo](https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz)
2. Install Dependencies

```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

3. Add the following to the `.bashrc` file:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```



4. Install all the required libraries with `conda`, using the `environment.yml` file:

```
conda env create --file environment.yml
```
