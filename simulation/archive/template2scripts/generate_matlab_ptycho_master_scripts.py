import os
import re


output_dir = "./p01_code/20231219_cluster_ptycho_recon_job_scripts"  # Replace with the desired output path
template_path = "./p01_code/runPtycho_master_template.m"

if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
        print(f'{output_dir} has been created!')

# Setup parameters

# data3D_200kV_df_20nm_ alpha_23.1mrad_ Cs_0.5um_ dp_256_ blur_0px_ dose_1.0e+08ePerAng2

# Script variables
script_name_prefix = 'runPtycho_master_'
cSAXS_matlab_path = '/home/fs01/cl2696/fold_slice_Yi' #'C:/Users/chiahao3/Box/01_Project_Analysis/2023_Q3_ML4Science/fold_slice_Lee' 

kmax_dirs = ['kmax2'] #['kmax2']
potential_modes = ['static'] #, 'phonon'
coherence_modes = ['coherent'] # 'coherent', 'temporal', 'spatial', temporal_spatial
alpha_list = ['alpha_1.0mrad','alpha_3.0mrad','alpha_5.0mrad','alpha_7.5mrad','alpha_9.5mrad','alpha_10.5mrad','alpha_12.0mrad','alpha_15.0mrad','alpha_18.0mrad','alpha_20.0mrad','alpha_22.0mrad','alpha_23.1mrad','alpha_24.0mrad','alpha_26.0mrad',
             'alpha_28.0mrad','alpha_30.0mrad','alpha_32.0mrad','alpha_34.0mrad','alpha_36.0mrad','alpha_38.0mrad','alpha_40.0mrad']

for kmax in kmax_dirs:
    for potential_mode in potential_modes:
        for coherence_mode in coherence_modes: 
                for alpha in alpha_list:
                    
                    # Read the content of the template file
                    with open(template_path, "r") as input_file:
                        lines = input_file.readlines()
                        
                    # Modify the specified lines
                    for i in range(len(lines)):
                        if "cSAXS_matlab_path= " in lines[i]:
                            lines[i] = f"cSAXS_matlab_path= '{cSAXS_matlab_path}/';\n"
                        elif "kmax_dir= " in lines[i]:
                            lines[i] = f"kmax_dir= '{kmax}';\n"
                        elif "potential_modes=" in lines[i]:
                            lines[i] = f"potential_modes= {{'{potential_mode}'}};\n"
                        elif "coherence_modes= " in lines[i]:
                            lines[i] = f"coherence_modes= {{'{coherence_mode}'}};\n"
                        elif "alpha_dirs= " in lines[i]:
                            lines[i] = f"alpha_dirs= {{'{alpha}_'}};\n"

                    output_fname = script_name_prefix + kmax + '_' + potential_mode + '_' + coherence_mode + "_" + alpha.replace('.','_') + ".m"
                    output_file_path = os.path.join(output_dir, output_fname)
                    # Write the modified content to the output file
                    with open(output_file_path, "w") as output_file:
                        output_file.writelines(lines)

                    print(f"{output_fname}' saved in '{output_dir}'.\n")