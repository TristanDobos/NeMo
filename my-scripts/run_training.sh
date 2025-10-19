#!bin/bash

init_conda
conda activate nemo
cd /mnt/matylda6/xdobos00/Nemo/examples/tts

# Default values
CONFIG_PATH="conf/audio_codec"
CONFIG_NAME="audio_codec_low_frame_rate_22050_nanocodec_nowavlm.yaml"

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
echo "Running: python ./audio_codec.py --config-path $CONFIG_PATH --config-name $CONFIG_NAME"
python ./audio_codec.py --config-path "$CONFIG_PATH" --config-name "$CONFIG_NAME"
