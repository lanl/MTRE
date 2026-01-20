#!/bin/bash

# List of all run scripts

scripts=(
"./MTRE/scripts/Triangle/run_LLaVA_NeXT.sh"
)


# Loop and create a temporary sbatch job file for each
for script in "${scripts[@]}"; do
    jobname=$(basename "$script" .sh)
    outfile="./logs/${jobname}.out"
    mkdir -p logs

    cat <<EOF > tmp_${jobname}.sbatch
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --output=${outfile}
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH -A xxxxx
#SBATCH --time=4:00:00
#SBATCH -C gpu80
#SBATCH --gres=gpu:1

module load python  # if your system uses modules

# Make sure Conda commands are available
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
conda activate MTRE_env

cd ./MTRE

bash ${script}
EOF

    sbatch tmp_${jobname}.sbatch
done
