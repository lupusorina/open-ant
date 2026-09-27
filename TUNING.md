# Hyperparameter tuning

> Three-stage Optuna pipeline producing two hyperparameter sets (hyperparams A for simulation, hyperparams B for hardware)

- **env A (`SimEmbodiedAntDR`):** Used for Stage 1 and Stage 2 training - randomized torso mass, floor friction, and actuator time constant
- **env B (`SimEmbodiedAnt`):** Used for Stage 1 eval, Stage 2 continual learning, and Stage 3 continual learning - best-effort digital twin of the real robot

## Stage 1: Broad sweep

Spawn parallel workers to run a broad hyperparameter sweep in simulation (env A)

```bash
python3 -m tuning.runner --entry agents.sac.tune_sac_stage1   --name sac_stage1   --journal stage1 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.mpo.tune_mpo_stage1   --name mpo_stage1   --journal stage1 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.mpo.tune_empo_stage1  --name empo_stage1  --journal stage1 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.mpo.tune_dmpo_stage1  --name dmpo_stage1  --journal stage1 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.mpo.tune_edmpo_stage1 --name edmpo_stage1 --journal stage1 --storage-dir runs/tuning --workers 8 --n-trials 1024
```

## Stage 2: Revalidation

Revalidate the top `--k` configs on fresh seeds to output `best_performer.json` (hyperparams A)

```bash
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study sac_stage1   --entry agents.sac.tune_sac_stage1   --out runs/tuning/stage2_sac   --k 8 --seeds 10 --workers 8
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study mpo_stage1   --entry agents.mpo.tune_mpo_stage1   --out runs/tuning/stage2_mpo   --k 8 --seeds 10 --workers 8
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study empo_stage1  --entry agents.mpo.tune_empo_stage1  --out runs/tuning/stage2_empo  --k 8 --seeds 10 --workers 8
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study dmpo_stage1  --entry agents.mpo.tune_dmpo_stage1  --out runs/tuning/stage2_dmpo  --k 8 --seeds 10 --workers 8
python3 -m tuning.revalidate --journal runs/tuning/stage1.journal --study edmpo_stage1 --entry agents.mpo.tune_edmpo_stage1 --out runs/tuning/stage2_edmpo --k 8 --seeds 10 --workers 8
```

## Stage 3: Continual fine-tune

Resume best performer to search continual learning space in env B (hyperparams B)

```bash
python3 -m tuning.runner --entry agents.continual --setup-arg runs/tuning/stage2_sac/best_performer.json   --name sac_continual   --journal stage3 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.continual --setup-arg runs/tuning/stage2_mpo/best_performer.json   --name mpo_continual   --journal stage3 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.continual --setup-arg runs/tuning/stage2_empo/best_performer.json  --name empo_continual  --journal stage3 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.continual --setup-arg runs/tuning/stage2_dmpo/best_performer.json  --name dmpo_continual  --journal stage3 --storage-dir runs/tuning --workers 8 --n-trials 1024
python3 -m tuning.runner --entry agents.continual --setup-arg runs/tuning/stage2_edmpo/best_performer.json --name edmpo_continual --journal stage3 --storage-dir runs/tuning --workers 8 --n-trials 1024
```

## Visualization

### Live Dashboard

```bash
optuna-dashboard runs/tuning/stage1.journal
```

### Report

```bash
python3 -m tuning.report --journal runs/tuning/stage1.journal --out runs/tuning/report_stage1.html --stage2-journal runs/tuning/stage1.journal
```
