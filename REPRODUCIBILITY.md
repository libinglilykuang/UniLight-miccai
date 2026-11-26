# DWKD implementation and reproducibility

The teacher-entropy variant implements Equations (1) and (2) as written in the Uni-Light manuscript: it uses temperature-scaled teacher probabilities and normalizes the entropy over the entire batch.

The preceding repository version used teacher–student cross-entropy with per-sample normalization. Its code remains available at [commit 444b7ae](https://github.com/libinglilykuang/UniLight-miccai/tree/444b7ae218934dd357cbe207282a0beef477a284).

The teacher-entropy variant has not yet been retrained or evaluated against the reported manuscript results. The exact historical training configuration and implementation used to obtain those results must be independently verified. Do not interpret this change as evidence that the published Dice, HD95, or other experimental results have been reproduced with the teacher-entropy variant.
