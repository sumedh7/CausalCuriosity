# CausalCuriosity
Official implementation of Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning at ICML 2021. 
[Paper](https://arxiv.org/abs/2010.03110) and [Website](https://sites.google.com/usc.edu/causal-curiosity/home)
## Installation

Download our version of [CausalWorld](https://github.com/rr-learning/CausalWorld) from this [drive](https://drive.google.com/drive/folders/1BWm0BuN8t3h9hJX-iA7Kp8q093Jub8fa?usp=sharing) link. Once downloaded add it to this repository and follow instructions to install [CausalWorld](https://github.com/rr-learning/CausalWorld).

You will also need [mujoco-py](https://github.com/openai/mujoco-py). Follow the installation instructions [here](https://github.com/openai/mujoco-py).
After installing mujoco-py, you will need to edit the done property for each of the mujoco agents property files. The ```done``` property needs to be set to ```False```. Otherwise the environment will stop simulating if the agents orientation exceeds a threshold. 

## Usage
For Mujoco experiments, run 
```python
python pnw_mujoco.py
```

For CausalWorld experiments, run
```python
python plan_and_write_video_vanilla_cw.py
```
##Citation
```
@article{sontakke2020causal,
  title={Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning},
  author={Sontakke, Sumedh A and Mehrjou, Arash and Itti, Laurent and Sch{\"o}lkopf, Bernhard},
  journal={arXiv preprint arXiv:2010.03110},
  year={2020}
}
```
