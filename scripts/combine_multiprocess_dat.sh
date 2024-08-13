#!/bin/bash

# Check if at least two arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 output_file file1 [file2 ...]"
    exit 1
fi

# The first argument is the output file
output_file=$1
shift # Shift the arguments so $@ now contains only the input files

# Clear the output file if it exists
>"$output_file"

# put the header in
head -n 1 $2 >>"$output_file"

# Loop through all the remaining arguments (input files)
for file in "$@"; do
    if [ -f "$file" ]; then
        tail -n +2 "$file" >>"$output_file"
        echo "" >>"$output_file" # Add a newline between files
    else
        echo "Warning: $file is not a valid file and will be skipped."
    fi
done

echo "Contents of all files have been written to $output_file."
