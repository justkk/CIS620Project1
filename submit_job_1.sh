# module load python/anaconda/3.5.6+tensorflow-gpu
# module load python/pytorch/1.6.0
module unload cuda
module load cuda/10.2

source activate /cbica/home/thodupv/.conda/envs/test1/
CUDA_VISIBLE_DEVICES=$(get_CUDA_VISIBLE_DEVICES) || exit
export CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/cbica/home/thodupv/.conda/envs/myenv/lib:$LD_LIBRARY_PATH
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
nvcc --version
#export PYTHONPATH=/cbica/home/thodupv/work/repos/harmonization:$PYTHONPATH
which python
python -c "import torch; print('hello'); print(torch.cuda.get_device_name(0)); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.rand(10).to(device)"


#PYTHONPATH=/cbica/home/thodupv/work/repos/harmonization python /cbica/home/thodupv/work/repos/harmonization/Experiments/TrainCycleGanGeneratorMatched.py --model_name="unet_128" --data_name="eve" --disc_loss="BCE" --cycle_loss="L1" --identity_loss="L1" --lambda_1=3.0 --lambda_2=3.0 --lambda_it=1.5

ifconfig


jupyter notebook --ip=0.0.0.0 --port=9000 --NotebookApp.token='' --NotebookApp.password=''

