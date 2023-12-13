### Environemnt

We provide a Dockerfile to build the docker containers we used in our experiments.

This can be built by going to the `segmentation_learning` directory and running
```
bash docker/build.sh
```
Then a container can be launched by running
```
bash docker/run.sh
```
which just runs a command and can be launched from anywhere, e.g., from the skill_learning directory in order to do the skill learning and planning.

The version of mujoco_py in this docker container is different than the one we originally used because of a bug that recently appeared involving Cython and mujoco_py: https://github.com/openai/mujoco-py/issues/773.
However, we don't believe this would have an effect on the simulation.

### Segmentation Learning

(Done in the `segmentation_learning` directory)

For the maze2d dataset, we generate our own versions of them because the downloadable d4rl large maze action data is far noisier than the policy generated by the d4rl provided script. We're uncertain of the cause of the noise in the downloadable action data.
We provide a lightly modified version of the d4rl script, `generate_maze2d_datasets.py`, that resets the environment when the episode time limit is hit and saves the episode in a different format.
For the kitchen data, we directly download the d4rl provided dataset.

The scripts that run our segmentation learning method are `maze2d_segmentation_rl_test.py` and `kitchen_segmentation_rl_test.py`.
The parameters for the models are handled by hydra in the `cfgs` directory.
Once segmentation models are learned, the `segmented_d4rl_dataset_generation.py` script can be used to fully annotate a dataset with $m_t$ inferred by the model.

### Skill Learning

(Done in the `skill_learning` directory)

Once we annotate a dataset with a segmentation model, we can move on to the skill learning phase.
To do so we use the `convert_maze2d_datset.py` and `convert_kitchen_dataset.py` to separate the learned segments and mask them so our skill encoder knows timesteps to encode while still being able to process variable length segments in parallel.
The fixed length variants of these scripts are for creating datasets for training OPOSM with the same architecture.
The skill learning training is done by the `point_mass_maze_experiment.py` and `kitchen_experiment.py` scripts.
Parameters are set in the YAML files in the `cfgs` directory.

### Planning

Planning is carried out by the `run_svlsm_planning.py` and `run_kitchen_planning.py` scripts. Parameters are given by command line arguments.