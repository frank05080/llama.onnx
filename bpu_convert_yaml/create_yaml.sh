#!/bin/bash

# Base configuration file
base_config="rwkv_mixing_0_config.yaml"

# Check if the base configuration file exists
if [ ! -f "$base_config" ]; then
  echo "Error: Base configuration file $base_config not found."
  exit 1
fi

# Loop from 2 to 23
for i in {2..23}
do
  # Format the new configuration filename
  new_config="rwkv_mixing_${i}_config.yaml"
  
  # Copy the base configuration file to the new file
  cp "$base_config" "$new_config"
  
  # Replace all occurrences of "mixing_0" with "mixing_xx"
  sed -i "s/mixing_0/mixing_${i}/g" "$new_config"
  
  echo "Created and modified $new_config"
done
