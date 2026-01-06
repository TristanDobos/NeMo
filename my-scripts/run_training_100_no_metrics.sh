#!/bin/bash
#$ -o /mnt/matylda6/xdobos00/runs/logs/202501051510-new.out            # standard output log file
#$ -e /mnt/matylda6/xdobos00/runs/logs/202501051510-new.err            # standard error log file

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

source /mnt/matylda6/xdobos00/nemo_final/bin/activate

export NEMO_DISABLE_ONE_LOGGER=1
export WANDB_MODE=offline

ulimit -n unlimited     # file descriptors
ulimit -u unlimited     # max user processes
ulimit -v unlimited     # virtual memory
ulimit -m unlimited     # resident set size
ulimit -s unlimited     # stack size
ulimit -l unlimited     # locked memory


# Default values
CONFIG_PATH="conf/audio_codec"
CONFIG_NAME="audio_codec_no_metrics.yaml"

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
  echo "  $0 --config-path conf/custom --config-name my_audio_config.yaml"
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

# Run the Python command
echo "Running: python /mnt/matylda6/xdobos00/NeMo/examples/tts/audio_codec.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME"
export CUDA_VISIBLE_DEVICES=$(~/scripts/free-gpus.sh 4)
python /mnt/matylda6/xdobos00/NeMo/examples/tts/audio_codec.py --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME" 



# hydra - konfigy hierarchicky!!!
# conf z nanocodecu


# submit to long queue
# pridat to wandb

# ak nie, replikovat bez low_framerate - povodne, malo by byt ovela rychlejsie
# ma sa konvergovat do par hodin - napisat mu, ak nie bez low frame