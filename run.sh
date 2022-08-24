#!/bin/bash
python3 run.py --mode=train --model_dir=./tmp/model_dir --config=configs/config_det_finetune.py --config.dataset.coco_annotations_dir=./colabs/annotations --config.train.batch_size=16 --config.train.epochs=3 --config.optimization.learning_rate=3e-5