### Generate Validation Datasets

There are 10 classes with 500 samples (subtomogram_mrc & json_label) per class in `./Datasets/train`

run `python generate_val.py` for random sampling 50 samples ((subtomogram_mrc & json_label)) per class into `./Datasets/val`

### Train Moco (main_moco.py)

```sh
# RB3D
python main_moco.py -a RB3D --lr 0.003 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 2 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.log

# DSRF3D_v2
python main_moco.py -a DSRF3D_v2 --lr 0.03 --schedule 300 400 500 --batch-size 32 --moco-k 128 --moco-dim 128 --workers 4 --epochs 600 --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --aug-plus ./Datasets/ 2>&1 | tee ./logs/arch-DSRF3D_v2_epochs600_bs32_lr0.03_moco-k128_moco-dim128.log
```

### Visualizaiton (main_cluster.py)

The visualization results and the extracted numpy features will be saved in `./Figures`.

> Notes: As long as the extracted numpy features exist in the `./Figures` path, the features will not be re-extracted

- Visualizing the `train` datasets:

    ```sh
    python main_cluster.py -a DSRF3D_v2 --lr 0.3 --batch-size 32 --workers 8 --epochs 100 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs120_bs32_lr0.3_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/
    ```

- Visualizing the `val` datasets:

    ```sh
    python main_cluster.py -a DSRF3D_v2 --lr 0.3 --batch-size 32 --workers 8 --epochs 100 --moco-dim 128 --pretrained ./main_moco_checkpoint/Nonorm-DSRF3D_v2_epochs120_bs32_lr0.3_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 --evaluate ./Datasets/
    ```

### Train a linear classifier (main_lincls.py)

```sh
python main_lincls.py -a RB3D --lr 30 --schedule 150 --batch-size 32 --workers 8 --epochs 200 --pretrained ./main_moco_checkpoint/arch-RB3D_epochs600_bs32_lr0.003_moco-k128_moco-dim128.pth.tar --dist-url "tcp://localhost:10001" --multiprocessing-distributed --world-size 1 --rank 0 --gpu 0 ./Datasets/ 2>&1 | tee ./logs/arch-RB3D_epochs200_bs32_lr30_moco-k128_moco-dim256_lincls_lr300.log
```

