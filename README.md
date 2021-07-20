# amdgpu
files for moco testing on the amd side (using gpu)

To train:
- make a main_moco_checkpoint folder in SSL_MacroMolecule_Real
- run using python3 main_moco.py  --batch-size=16 --dist-url ‘tcp://localhost:10001’ --multiprocessing-distributed --world-size 1 --rank 0 “/path/to/training/data"
- checkpoints will save there

To test / cluster:
- download my trained checkpoint final weights into main_moco_checkpoint via this drive link: 
https://drive.google.com/drive/folders/1drxNnh5_YxyrYTjEzmRpPwp6_X9C1vQo?usp=sharing
- untar temps_10.tar and inf_10.tar for template directory and test data directory, respectively
- edit downstream.py (testdir and tempdir) to match your filepaths for temps_10 and inf_10
- run using python3 downstream.py --batch-size=16 --dist-url ‘tcp://localhost:10001’ --multiprocessing-distributed --world-size 1 --rank 0 “/path/to/training/data/not/used" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar 
