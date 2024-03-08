import os
import numpy as np

output_dir = "./p01_code/20231219_cluster_prepare_hdf5_job_scripts"  # Replace with the desired output path
template_path = "./p01_code/prepare_ptycho_hdf5_template.py"

potential_modes= ['static'] # 'phonon'
coherence_modes= ['coherent','temporal_spatial'] # 'coherent', 'temporal', 'spatial', temporal_spatial
alpha_dirs= ['alpha_1.0mrad','alpha_3.0mrad','alpha_5.0mrad','alpha_7.5mrad','alpha_9.5mrad','alpha_10.5mrad','alpha_12.0mrad','alpha_15.0mrad','alpha_18.0mrad','alpha_20.0mrad','alpha_22.0mrad','alpha_23.1mrad','alpha_24.0mrad','alpha_26.0mrad',
             'alpha_28.0mrad','alpha_30.0mrad','alpha_32.0mrad','alpha_34.0mrad','alpha_36.0mrad','alpha_38.0mrad','alpha_40.0mrad'] 
cs_dirs= ['Cs_1.0um', 'Cs_1.1mm']
# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through desire variants

for potential_mode in potential_modes:
    for coherence_mode in coherence_modes:
        for alpha_dir in alpha_dirs:
            for cs_dir in cs_dirs:
                # Read the content of the template file
                with open(template_path, "r") as input_file:
                    lines = input_file.readlines()

                # Modify the specified lines
                for i in range(len(lines)):
                    if "potential_modes= " in lines[i]:
                        lines[i] = f"potential_modes= ['{potential_mode}']\n"
                    elif "coherence_modes= " in lines[i]:
                        lines[i] = f"coherence_modes= ['{coherence_mode}']\n"
                    elif "alpha_dirs= " in lines[i]:
                        lines[i] = f"alpha_dirs= ['{alpha_dir}']\n"
                    elif "cs_dirs= " in lines[i]:
                        lines[i] = f"cs_dirs= ['{cs_dir}']\n"

                output_fname = f"prepare_ptycho_hdf5_{potential_mode}_{coherence_mode}_{alpha_dir}_{cs_dir}.py"
                output_file_path = os.path.join(output_dir, output_fname)
                # Write the modified content to the output file
                with open(output_file_path, "w") as output_file:
                    output_file.writelines(lines)

                print(f"File '{template_path}' processed and saved as '{output_fname}' in '{output_dir}'.")
