#!/bin/bash
#$ -o /mnt/matylda6/xdobos00/runs/logs/focalcodec-3.out            # standard output log file
#$ -e /mnt/matylda6/xdobos00/runs/logs/focalcodec-3.err            # standard error log file


ulimit -f unlimited -t unlimited -v unlimited -s unlimited -n $(ulimit -Hn)

init_conda() {
 __conda_setup="$('/mnt/matylda6/xdobos00/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
 if [ $? -eq 0 ]; then
     eval "$__conda_setup"
 else
     if [ -f "/mnt/matylda6/xdobos00/miniconda/etc/profile.d/conda.sh" ]; then
         . "/mnt/matylda6/xdobos00/miniconda/etc/profile.d/conda.sh"
     else
         export PATH="/mnt/matylda6/xdobos00/miniconda/bin:$PATH"
     fi
 fi
 unset __conda_setup
}

# source /mnt/matylda6/xdobos00/nemo_final/bin/activate
. /mnt/matylda6/xdobos00/nemo_final/bin/activate

export NEMO_DISABLE_ONE_LOGGER=1
export WANDB_MODE=offline



# Default values
CONFIG_PATH="conf/audio_codec"
CONFIG_NAME="audio_codec_low_frame_rate_22050_focalcodec_slow.yaml"

# Function to display help
show_help() {
  echo "Usage: $0 [--config-path <path>] [--config-name <name>] [--help]"
  echo
  echo "Runs the audio codec Python script with optional parameters."
  echo
  echo "Options:"
  echo "  --config-path <path>   Path to the config directory (default: $CONFIG_PATH)"
  echo "  --config-name <name>   Name of the config file (default: $CONFIG_NAME)"
  echo "  --help                 Show this help message and exit"
  echo
  echo "Example:"
  echo "  $0 --config-path conf/custom --config-name audio_codec_low_frame_rate_22050_focalcodec.yaml"
  exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --config-name)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options."
      exit 1
      ;;
  esac
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python command
echo "Running: python /mnt/matylda6/xdobos00/NeMo/examples/tts/audio_codec.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME"
export CUDA_VISIBLE_DEVICES=$(~/scripts/free-gpus.sh 4)
python /mnt/matylda6/xdobos00/NeMo/examples/tts/audio_codec.py --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" 


# hydra - konfigy hierarchicky!!!
# conf z nanocodecuqsub -N focalcodec -q long.q -l gpu=4,gpu_ram=20G,ram_free=20G,mem_free=20G,matylda6=1 /mnt/matylda6/xdobos00/NeMo/my-scripts/run_training_100_w_metrics_paper_focalcodec.sh


# submit to long queue
# pridat to wandb

# ak nie, replikovat bez low_framerate - povodne, malo by byt ovela rychlejsie
# ma sa konvergovat do par hodin - napisat mu, ak nie bez low frame

# sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/ && sleep 300 &&  wandb sync /homes/eva/xd/xdobos00/.wandb_osh_command_dir/wandb/offline-run-20260106_194446-xhkchwl2/

# qsub -N focalcodec -q all.q -l gpu=4,gpu_ram=20G,ram_free=20G,mem_free=20G,matylda6=1 /mnt/matylda6/xdobos00/NeMo/my-scripts/run_training_100_w_metrics_paper_focalcodec.sh


#  The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=31` in the `DataLoader` to improve performance.



