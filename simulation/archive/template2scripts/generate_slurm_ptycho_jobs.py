import os

# Template for the Slurm script
slurm_template = """#!/bin/bash
#SBATCH --job-name={slurm_job_name}
#SBATCH --mail-user=cl2696@cornell.edu  # Where to send mail
#SBATCH --nodes=1                      # number of nodes requested
#SBATCH --ntasks=1                     # number of tasks to run in parallel
#SBATCH --cpus-per-task=4              # number of CPUs required for each task
#SBATCH --gres=gpu:2g.20gb:1           # request a GPU
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --output=log_job_%j_{slurm_job_name}_{slurm_job_identifier}.txt  # Standard output and error log

pwd; hostname

# Clear the value set in the DISPLAY environment variable
# to run the CLI version of MATLAB
unset DISPLAY

# Load MATLAB module (Enable MATLAB in user environment)
module load matlab/R2021a
module load cuda/11.5

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"

matlab -nodisplay -nosplash -r "{cluster_script_name}; exit" 2>&1

date


# Don't use dash (-) in the <matlab_script_name> because '-r' will fail to recognize the script name.
# The '.m' file extension is not needed for <matlab_script_name> while using '-r'
"""

# Directory containing Ptycho scripts
local_script_dir          = "p01_code/20231219_cluster_ptycho_recon_job_scripts/"
local_script_name_pattern = "runPtycho_master_"
slurm_script_name_pattern = "slurm_matlab_runPtycho_"
slurm_job_name            = "ptycho"

# Enumerate Ptycho scripts in the directory
for script_name in os.listdir(local_script_dir):
    if script_name.startswith(local_script_name_pattern) and script_name.endswith(".m"):
        # Extract the identifier part from the filename
        identifier_chars = script_name[len(local_script_name_pattern):-len(".m")]
        
        # Create the corresponding Slurm script content
        slurm_content = slurm_template.format(slurm_job_name = slurm_job_name,\
                                              slurm_job_identifier=identifier_chars,\
                                              cluster_script_name=script_name.strip(".m"))

        # Save the Slurm script with a name based on the identifier characters
        slurm_filename = slurm_script_name_pattern + identifier_chars + ".sub"
        slurm_job_path = os.path.join(local_script_dir, slurm_filename)
        with open(slurm_job_path, "w", newline='\n') as slurm_file:
            slurm_file.write(slurm_content)

        print(f"Generated Slurm script: {slurm_filename}")