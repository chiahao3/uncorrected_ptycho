import os
import numpy as np

output_dir = "./p01_code/20231219_cluster_abTEM_job_scripts"  # Replace with the desired output path
template_path = "./p01_code/run_abTEM_template.py"

num_jobs = 32 # Altas would only accept 30 running jobs per user
num_indices = 1024
start_indices = np.int32(np.arange(0, num_indices, np.ceil(num_indices/num_jobs)))
end_indices = np.int32(np.array((*start_indices[1:], num_indices)))

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through desire variants

for idx in range(num_jobs):

    # Read the content of the template file
    with open(template_path, "r") as input_file:
        lines = input_file.readlines()

    # Modify the specified lines
    for i in range(len(lines)):
        if "start_index= " in lines[i]:
            lines[i] = f"start_index= {start_indices[idx]}\n"
        elif "end_index= " in lines[i]:
            lines[i] = f"end_index= {end_indices[idx]}\n"

    output_fname = f"run_abTEM_realistic_ptycho_simu_{str(idx+1).zfill(2)}.py"
    output_file_path = os.path.join(output_dir, output_fname)
    # Write the modified content to the output file
    with open(output_file_path, "w") as output_file:
        output_file.writelines(lines)

    print(f"File '{template_path}' processed and saved as '{output_fname}' in '{output_dir}'.")
