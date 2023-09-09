#!/bin/bash

# Your Python script file (replace with the actual file path)
PYTHON_SCRIPT="main.py"

# Maximum number of simultaneous runs
MAX_SIMULTANEOUS_RUNS=5

# Function to monitor and limit the number of background processes
monitor_processes() {
  while true; do
    # Count the number of background processes
    num_processes=$(jobs | wc -l)

    # If the number of background processes is less than the maximum, break the loop
    if [ "$num_processes" -lt "$MAX_SIMULTANEOUS_RUNS" ]; then
      break
    fi

    # Sleep for a while before checking again
    sleep 1
  done
}

# Loop from 1 to 50 and run the Python script in the background
for ((arg=1; arg<=10; arg++)); do
  # Monitor and limit the number of background processes
  monitor_processes

  # Run the Python script with the current integer argument in the background
  python "$PYTHON_SCRIPT" "$arg" &
  echo "Running $PYTHON_SCRIPT with argument: $arg"
done

# Wait for all background processes to finish
wait
echo "All processes have completed."