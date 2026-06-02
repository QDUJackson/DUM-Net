# 26_525 T=7 Elasticity Model Package

This folder contains only the code path used by `26_525/best_dice_model_swin.pth`,
plus the `data8k10` dataset and the required weights.

## Contents

- `26_525/best_dice_model_swin.pth`: main T=7 model weight.
- `Vxm_Path/best_dice_model_vxm.pth`: pre-registration weight required during model construction.
- `data8k10/`: train/val/test data.
- `module/`: required model modules for `UTN_More_Layers_Init_Lambda.UTNet`.
- `MIDataSet.py`: dataset loader.
- `losses.py`: training/evaluation losses.
- `test_26_525.py`: evaluate Dice on `data8k10`.
- `train_26_525.py`: train or finetune the same model structure.

## Environment

Use the local `pytorch05` environment:

```powershell
& C:\Users\24088\.conda\envs\pytorch05\python.exe test_26_525.py
```

## Evaluate

Run from this folder:

```powershell
cd C:\Jac5On\CS_NEW323\release_26_525
& C:\Users\24088\.conda\envs\pytorch05\python.exe test_26_525.py
```

Report each unfolded layer's Dice:

```powershell
& C:\Users\24088\.conda\envs\pytorch05\python.exe test_26_525.py --layer-dice
```

Expected final Dice on `data8k10/val` is about `0.85958`.

## Train Or Finetune

Train from scratch:

```powershell
& C:\Users\24088\.conda\envs\pytorch05\python.exe train_26_525.py
```

Finetune from the packaged weight:

```powershell
& C:\Users\24088\.conda\envs\pytorch05\python.exe train_26_525.py --init-weights 26_525\best_dice_model_swin.pth --out-dir outputs\finetune_26_525
```

Resume training:

```powershell
& C:\Users\24088\.conda\envs\pytorch05\python.exe train_26_525.py --resume outputs\train_26_525\latest.pth
```
