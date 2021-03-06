# GalilAI: Out-of-Task Distribution Detection using Causal Active Experimentation for Safe Transfer RL

## Causal World experiments

Run causal world experiments on (size, mass) settings. 
You may also modify `cem_planner_cw.py` and `cem_planner_cw_baseline.py` to
test more pairs.

1. run `cem_planner_cw.py` to save OOD detection with causal curiosity results under `./pickle/CW/SizeMass`
2. run `cem_planner_cw_baseline.py` to save OOD detection baseline results under `./pickle/CW/SizeMass/SA`
3. run `cem_planner_cw_res.py` to show results for 1 and 2 in matrix form, which are shown as heatmaps in the paper


## Mujoco experiments

A sample workflow to run all code for a given experiment is shown in `run_all_mujoco.sh`.
You may modify this scipt to suit your needs. 
To see a list of avaliable options for any script, run `python [script_name].py --help`.

The workflow consists of 5 parts.
1. Run GalilAI method
    * run the `plan_mujoco.py` script with desired flags.
2. Collect additional (state, action) pairs for baseline experiments.
    * run `plan_mujoco.py` again, now with `--baseline_only True` and making sure you specify the same output path as before.
3. Train PNNs to get mean & covariance vectors for baselines. 
    * run `train_pnn.py`.
4. Calculate KL divergence between (unseen, seen) causal pairs. 
    * run `train_pnn.py` again with same flags, adding `--compare_only true`
5. Plot results!
    * for this, please see `plot.py` and modify the directory names accordingly.
        
## Reference

If you would like to cite our paper,
> Sumedh A Sontakke*, Stephen Iota*, Zizhao Hu*, Arash Mehrjou, Laurent Itti and Bernhard Schölkopf. GalilAI: Out-of-Task Distribution Detection using Causal Active Experimentation for Safe Transfer RL. Accepted to *The 25th International Conference on Artificial Intelligence and Statistics* (AISTATS), 2022. [[arXiv](https://arxiv.org/abs/2110.15489)].

```       
@article{sontakke2021galilai,
    author = {Sumedh A Sontakke and Stephen Iota and Zizhao Hu and Arash Mehrjou and Laurent Itti and Bernhard Schölkopf},
    title = {GalilAI: Out-of-Task Distribution Detection using Causal Active Experimentation for Safe Transfer RL},
    year = {2021},
    eprint = {arXiv:2110.15489},
    url = {https://arxiv.org/abs/2110.15489},
    note = {Accepted to \textit{The 25th International Conference on Artificial Intelligence and Statistics} (AISTATS), 2022}
}
```
