[![Tests](https://github.com/dmalynyak/ИМЯ_РЕПОЗИТОРИЯ/actions/workflows/tests.yml/badge.svg)](https://github.com/dmalynyak/depth-estimator-pytorch/actions/workflows/tests.yml)

get nyu datasets:
```bash
datasets 
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
wget http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
```

weights:
```bash
wget https://github.com/dmalynyak/depth-estimator-pytorch/releases/tag/nyu_1/nyu_weights.pt -O weights/nyu.pt
```

train:
```bash
python -m src.train --device cuda --chkpt_path excess/models/best.pt --log_path excess/models/metrics.csv
tensorboard --logdir="excess/models/metrics.csv" 
```

inference:
```bash
python -m nyu_inference --device cuda --file_path excess/oleg.jpg --model_path excess/models/best_02.pt 
```