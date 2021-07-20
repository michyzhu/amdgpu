ls
vim Custom_CryoET_DataLoader.py 
ls
cd SSL_MacroMolecule_Real
ls
vim main_moco.py
ls
cd ..
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
ls
logout
ls
cd data3_SNRinfinity
ls
cd densitymap_mrc
ls
cd ..
cd json_
ls
cd json_label/
ls
cat target2930.json
cat target1.json
cat target2.json
cat target50.json
cat target100.json
cat target500.json
cat target250.json
cat target499.json
cat target500.json
cat target5000.json
cat target4999.json
ls
cat target4999.json
cat target4499.json
cat target3499.json
cat target2499.json
cat target1499.json
cat target499.json
cat target500.json
cat target1000.json
cat target1500.json
cat target2500.json
cat target3500.json
cat target4500.json
ls
cd ..
ls
cd densitymap_mrc
ls
cd ..
ls
cd subtomogram_mrc
ls
cd ..
ls
cd SSL_MacroMolecule_Real
ls
vim Custom_cr
vim Custom_CryoET_DataLoader.py 
ls
vim main_moco
vim main_moco.py
cd ..
ls
cd data3_SNRinfinity
ls
cd densitymap_
cd densitymap_mrc
ls
clear
ls
cd data3_SNRinfinity
ls
cd densitymap_mrc
ls
cd ..
ls
cd json_label
ls
cat target2793.json
clear
ls
cd ..
ls
clear
ls
cd ..
ls
pwd
ls
cd data3_SNRinfinity
ls
cd subtomogram_mrc
ls
cd ..
ls
cd ..
ls
tar -xvf myzData10.tar
ls
ls data3_SNRinfinity
mv json jsonpack packmap myzData10
mkdir myzData10
mv json jsonpack packmap myzData10
ls
cd myzData10
ls
cd ..
mv tomo test map_single myzData10
ls
cd myzData10
ls
mv json json_label
mv tomo/target subtomogram_mrc
ls
cd tomo
ls
cd target
ls
cd ..
cd myzData10
cd subtomogram_mrc/
ls
cd ..
mv subtomogram_mrc/mrc subtomogram_mrc/
ls
cd subtomogram_mrc/
ls
cd ..
rm -d tomo
cd subtomogram_mrc/
ls
rm -d png
cd mrc
ls
cd ..
mv *.mrc .
mv "*.mrc" .
mv -r "*.mrc" .
cd mrc
mv *.mrc ../
ls
cd ..
ls
rm mrc
rm -d mrc
ls
cd ..
ls
cd ..
clear
ls
cd SSL_MacroMolecule_Real
ls
vim main_moco.py
ls
vim main_moco.py
ls
vim main_moco.py
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity"
cd ..
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
cd SSL_MacroMolecule_Real
ls
vim main_moco
vim main_moco.py
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
cd SSL_MacroMolecule_Real
ls
vim main_moco.py 
ls
cd main_moco_checkpoint/
ls
cd ~
ls
cd myzData10
ls
cd json
cd json_label/
ls
cat target0.json
cd ..
ls
cd subtomogram_mrc/
ls
pwd
ls
cd SSL_MacroMolecule_Real
ls
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
htop
singularity shell ./pytorch_latest.sif 
cd SSL_MacroMolecule_Real
ls
vim main_moco
vim main_moco.py
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
vim main_moco.py
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd ..
ls
squeue -u myz
cd data3_SNRinfinity
ls
cd ..
cd myzData10
ls
cd 4d4r
ls
pwd
ls
vim check.py
ls
cd ..
ls
cd data3_SNRinfinity
ls
cd densitymap_mrc
ls
cd ../json_label/
ls
cat target0.json
ls
cd ..
ls
cd ..
ls
tar -xvf inf_10.tar
ls
mv snrinf inf_10
ls
cd inf_10
ls
cd $VIM
ls
cd inf_10
ls
cd 1bxn
cd ..
ls
vim conv.py
cd 1bxn
ls
cat target0.json
cd ..
vim conv.py
python3 conv.py
vim conv.py
python3 conv.py
vim conv.py
python3 conv.py
ls
rm "*.mrc"
ls
cd 1bxn
ls
cd ..
python3
vim conv.py
python3 conv.py
vim conv.py
python3 conv.py
cd 1yg6
ls
cd ..
ls
mv moltarget36.json target36.json
mv moltomo3.mrc tomo3.mrc
mv moltomo39.mrc tomo39.mrc
mv moltomo5.mrc tomo5.mrc
ls
mv target36.json 1yg6
mv tomo3.mrc 1yg6
mv tomo39.mrc 1yg6
mv tomo5.mrc 1yg6
ls
cd 1yg6
ls
cd ..
python3 conv.py
vim conv.py
python3 conv.py
ls
cd 1bxn
ls
cd ..
vim conv.py
python3 conv.py
cd 1yg6
ls
cd ..
ls
find . -name "*.mrc" -exec mv -i {} -t subtomogram_mrc \;
mkdir subtomogram_mrc
find . -name "*.mrc" -exec mv -i {} -t subtomogram_mrc \;
ls
cd 1bxn
ls
cd ..
mkdir json_label
find . -name "*.json" -exec mv -i {} -t json_label \;
ls
cd 1bxn
ls
cd ..
find . -name "1*" -type f -delete
ls
rm -d 1bxn 1f1b 1yg6 2byu
ls
rm -d 2h12  2ldb  3gl1  4d4r/6t3e
rm -d 2h12  2ldb  3gl1  4d4r 6t3e
ls
cd ..
ls
cd SSL_MacroMolecule_Real
cd ..
ls
cd snrinf
ls
cd inf_10
ls
cd ..
squeue -u myz
ls
singularity shell ./pytorch_latest.sif 
cd SSL_MacroMolecule_Real
ls
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd ..
singularity shell ./pytorch_latest.sif 
clear
ls
vim check.py
ls
cd data3_SNRinfinity
ls
cd densitymap_mrc/
ls
pwd
cd ..
ls
rm -d myzData10
rm -r myzData10
tar -xvf myzData10.tar 
ls
mv test myzData10
ls
cd myzData10
kls
ls
cd infinite/
ls
cd ..
rm -d infinite/
ls
cd ..
ls
cd SSL_MacroMolecule_Real
ls
vim Custom_CryoET_DataLoader.py 
cd ..
ls
cd myzData10
ls
mkdir json_label
mkdir subtomogram_mrc
ls
cd 4d4r
ls
cd ..
ls
mmv
rename
ls
cd 1bxn
rename -n 's/target/1bxntarget/' *
rename 's/target/1bxntarget/' *
ls
rename 's/target/1bxntarget/' *
ls
mkdir test
cd test
ls
vim abcd.txt
ls
rename 's/abc/xyz/' *
rename "s/abc/xyz/" *
rename 's/abc/xyz/' *
rename . 's/abc/xyz/' *
rename 's/abc/xyz/' .
rename 's/abc/xyz/' . .
rename 's/abc/xyz/' . *
ls
rename 's/abc/xyz/' \*.txt
rename(1)
rename
rename -h
rename abc xyz \*.txt
ls
rename abc xyz *
ls
cd ..
rename target 1bxntarget *
ls
rm -d test
rm -r test
rename target 1bxntarget * && rename tomo 1bxntomo *
ls
cd ..
ls
cd 1f1b
rename target 1f1btarget * && rename tomo 1f1btomo *
cd ../1yg6
rename target 1yg6target * && rename tomo 1yg6tomo *
cd ../2byu
rename target 2byutarget * && rename tomo 2byutomo *
cd ../2h12 && rename target 2h12target * && rename tomo 2h12tomo *
cd ../2ldb && rename target 2ldbtarget * && rename tomo 2ldbtomo *
cd ../3gl1 && rename target 3gl1target * && rename tomo 3gl1tomo *
cd ../4d4r && rename target 4d4rtarget * && rename tomo 4d4rtomo *
cd ../6t3e && rename target 6t3etarget * && rename tomo 6t3etomo *
cd ..
ls
find . -name "*target*.json" -exec mv -i {} -t json_label \;
ls
cd json_label/
ls
python3
ls
cd ..
find . -name "*tomo*.mrc" -exec mv -i {} -t subtomogram_mrc \;
ls
cd subtomogram_mrc/
sl
ls
cd ..
cd SSL_MacroMolecule_Real
ls
vim main_moco.py
cd ..
module load singularity/3.7.2 
singularity shell ./pytorch_latest.sif 
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
ls
cd SSL_MacroMolecule_Real
ls
vim main_moco.py
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
ls
vim main_moco.py
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
clear
ls
cd data3_SNRinfinity
ls
cd ..
cd SSL_MacroMolecule_Real
ls
vim main_moco.py
vim Custom_CryoET_DataLoader.py 
cd ../myzData10
ls
htop
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
squeue -u myz
rocm-smi
scancel 23382
squeue -u myz
scancel 23383
cd ../SSL_MacroMolecule_Real
ls
vim main_moco.py
vim Custom_CryoET_DataLoader.py 
cd ..
module load singularity/3.7.2 
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
squeue -u myz
ls
mv inf_10/conv.py .
ls
rm inf_10.tar
mv myzData10.tar
rm temps10.tar
rm myzData10.tar
ls
tar -xvf temps10.tar
ls
rm temps10.tar
mv temps temps10
cd temps10
ls
cd ..
vim conv.py
python3 conv.py
vim conv.py
python3 conv.py
ls
cd temps10
ls
cd 1bxn
ls
cd ..
mkdir subtomogram_mrc
find . -name "*.mrc" -exec mv -i {} -t subtomogram_mrc \;
mkdir json_label
find . -name "*.json" -exec mv -i {} -t json_label \;
ls
cd json_label
ls
python3
cd ../subtomogram_mrc/
python3
ls
cd ..
ls
cd ..
ls
cd inf_10
ls
cd subtomogram_mrc
ls
python3
ls
cd ..
rm -r .
rm -r ./
ls
rm -r json_label
rm -r subtomogram_mrc
cd ..
rm -d inf_10
ls
tar -xvf inf_10.tar
ls
mv snrinf inf_10
cd inf_10
ls
cd ..
vim conv.py 
python3 conv.py
cd inf_`
'

cd inf_10
ls
cd 1bxn
ls
cd ..
mkdir subtomogram_mrc
find . -name "*.json" -exec mv -i {} -t json_label \;
mkdir json_label
find . -name "*.json" -exec mv -i {} -t json_label \;
find . -name "*.mrc" -exec mv -i {} -t subtomogram_mrc \;
ls
cd subtomogram_mrc
ls
python3
cd ..
ls
rm -d 1bxn  1f1b  1yg6  2byu  2h12  2ldb  3gl1  4d4r  6t3e
ls
cd ..
cd temps10/
ls
rm -d 1bxn  1f1b  1yg6  2byu  2h12  2ldb  3gl1  4d4r  6t3e
cd ..
l;s
ls
cd SSL_MacroMolecule_Real
ls
squeue -u myz
ls
vim main_moco.py
cd SSL_MacroMolecule_Real
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd moco
ls
vim builder.py 
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd ..
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
pwd
ls
vim Custom_CryoET_DataLoader.py 
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
vim Custom_CryoET_DataLoader.py 
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd moco
ls
 vim builder.py
cd ..
vim Custom_CryoET_DataLoader.py 
cd moco
vim loader.py 
cd ..
vim Custom_CryoET_DataLoader.py 
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
netstat -nltp
fuser 
lsof
lsof -i:10003
npx
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
singularity shell ./tensorflow_latest.sif 
singularity shell ./pytorch_latest.sif 
ls
cd SSL_MacroMolecule_Real
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd ..
ls
cd temps10
ls
cd ..
cd SSL_MacroMolecule_Real
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd ../temps10/
ls
cd json_label
ls
cd ../subtomogram_mrc/
ls
cd ..
ls
cd inf
cd inf_10
ls
cd subtomogram_mrc/
ls
cd ..
ls
cd temps10
ls
cd json_label/
ls
cd ..
mkdir json_packing
mv json_label/*target*.json json_packing
ls
cd json_
cd json_packing/
ls
cd ..
ls
mv json_label json_pack
mv json_packing json_label
ls
cd json_label
ls
cd ..
mkdir tomo_pack
ls
mv subtomogram_mrc/*packing*.mrc tomo_pack
ls
cd subtomogram_mrc/
ls
cd ..
mv subtomogram_mrc/*map*.mrc tomo_pack
cd subtomogram_mrc/
ls
cd ..
cd 3hhb
ls
cd ..
rm -d 3hhb
cd json_label
ls
cd ..
mv json_label/3hhb*.mrc .
mv json_label/3hhb* .
ls
cd json_label
ls
cd ..
ls
cd ..
ls
cd SSL_MacroMolecule_Real
ls
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
clear
cd SSL_MacroMolecule_Real
vim main_moco
vim main_moco.py
vim Custom_CryoET_DataLoader.py 
vim main_moco.py
vim Custom_CryoET_DataLoader.py 
vim main_moco.py
vim Custom_CryoET_DataLoader.py 
vim main_moco.py
vim downstream.py 
ls
vim Custom_CryoET_DataLoader.py 
cd ..
ls
tar -xvf 3hhbinf.tar 
ls
rm 3hhbinf.tar
cd 3hhb
ls
cd ..
ls
mv 3hhb/target*.json inf_10/json_label
cd 3hhb
ls
cd ..
cd inf_10
ls
cd json_label/
ls
cd ..
vim conv.py
ls
cd ..
vim conv.py 
python3 conv.py 
vim conv.py 
python3 conv.py 
vim conv.py 
python3 conv.py 
ls
cd inf_10
ls
cd subtomogram_mrc/
ls
cd ..
vim conv.py 
python3 conv.py 
vim conv.py 
python3 conv.py 
vim conv.py 
python3 conv.py 
ls
mv inf_10 f10
tar -xvf inf_10.tar
ls
cd snrinf
ls
cd ..
vim conv.py 
python3 conv.py
ls
cd snrinf
ls
cd 1bxn
ls
cd ..
mv snrinf/*/*target*.json f1f0/json_label
mv snrinf/*/*target*.json f10/json_label
cd f10/json_label/
ls
rm target*.json
ls
python3
cd ..
cd subtomogram_mrc/
ls
python3
cd ..
ls
cd ..
ls
cd 3hhb
ls
rm map*
ls
rm packing*
ls
cd ..
ls
cd snrinf
ls
cd ..
cd f10
ls
cd json_label/
ls
cd ..
ls
tar -xvf 3hhbinf.tar 
ls
cd 3hhb
ls
rm map*
rm packing*
ls
cd ..
vim conv.py 
python3 conv.py
cd 3hhb
ls
cd ..
mv 3hhb/*.json snrinf/json_label
mv 3hhb/*.json f10/json_label
mv 3hhb/*.mrc f10/subtomogram_mrc/
cd 3hhb
ls
cd ..
cd f10
ls
cd subtomogram_mrc/
ls
python3
cd ../json_label/
python3
ls
cd ..
ls
cd ..
ls
mv f10 inf_10
rm -r 3hhb
rm 3hhbinf.tar 
rm -r snrinf
ls
rm -r myzData10/
ls
tar cvf inf10myz.tar inf_10
ls
cd SSL_MacroMolecule_Real
clear
ls
cd ..
ls
cd temps10
ls
mv 3hhb* json_label
ls
cd json_label
ls
cd ..
cd subtomogram_mrc/
ls
cd ..
ls
cd ..
ls
tar xvf 3hhb.tar 
ls
vim conv.py 
python3 conv.py 
cd 3hhb
ls
cd ..
mv 3hhb/3hhbtomo*.mrc temps10/subtomogram_mrc/
cd temps10
ls
cd subtomogram_mrc/
ls
python3
cd ..
ls
rm 3hhb.tar
rm -r 3hhb
ls
rm inf10myz.tar 
ls
cd SSL_MacroMolecule_Real
ls
clear
ls
vim downstream.py 
ls
cd SSL_MacroMolecule_Real
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd moco
ls
vim builder.py 
cd ..
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd moco
ls
vim builder.py 
ls
cd ..
ls
cd ENc
cd Encoder3D/n
cd Encoder3D/
ls
vim Model_RB3D.py
cd ../moco
vim builder.py 
cd ..
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 downstream.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
cd ..
ls
cd inf_10
ls
cd json_label
ls
cd ..
cd subtomogram_mrc/
ls
cd ../json_label/
la
ls
cd ..
ls
cd ..
ls
cd inf_10
cd subtomogram_mrc/
ls
cd ..
ls
cd ..
ls
cd SSL_MacroMolecule_Real
clear
ls
cd moco
ls
vim builder.py 
cd ../Encoder3D/
ls
vim Model_RB3D.py 
cd ../moco
vim builder.py 
cd ..
ls
cd ..
srun --nodes=1 --ntasks-per-node=20 --cpus-per-task=1  --gres gpu:8 --pty bash -i
ls
cd SSL_MacroMolecule_Real
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
singularity shell ./pytorch_latest.sif 
cd SSL_MacroMolecule_Real
ls
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 main_moco_v2.py   -a RB3D   --lr 0.03   --batch-size=16 --moco-k=128   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --aug-plus
ls
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd ..
ls
cd SSL_MacroMolecule_Real
ls
cd ..
cd data3_SNRinfinity
ls
cd densitymap_mrc
ls
cd ..
ls
cd ..
ls
cd SSL_MacroMolecule_Real
ls
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd Encoder3D/
ls
vim Model_RB3D.py 
vim down
ls
cd ..
vim downstream.py 
cd moco
ls
vim builder.py 
cd ../Encoder3D/
ls
vim Model_RB3D.py 
cd ..
ls
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
python3 downstream.py   --lr 0.03   --batch-size=16 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  "/home/myz/data3_SNRinfinity" --resume=main_moco_checkpoint/checkpoint_0199.pth.tar
ls
cd Encoder3D/
ls
vim Model_RB3D.py 
vim Model_SCNN.py 
vim Model_DSRF3D_v2.py 
ls
cd ..
ls
cd Encoder3D/
ls
vim Model_RB3D.py 
cd ..
