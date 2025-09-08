import os
# Current working directory
curr_dir = os.path.dirname(os.path.abspath(__file__))
curr_dir = os.path.abspath(os.path.join(curr_dir, '..'))


# Define main directories
repo_dir = os.path.join(curr_dir, "repo/onclass/repo")
data_dir = os.path.join(curr_dir, "repo/onclass/data")

# Subdirectories for data
scrna_data_dir = os.path.join(data_dir, 'OnClass_data_public', 'my_data/')
ontology_data_dir = os.path.join(data_dir, 'OnClass_data_public', 'Ontology_data/')
intermediate_dir = os.path.join(data_dir, 'OnClass_data_public', 'Intermediate_files/')

# Output and figure directories
result_dir = os.path.join(repo_dir, 'result', 'SingleCell', 'OnClass', 'Reproduce')
figure_dir = os.path.join(repo_dir, 'result', 'SingleCell', 'OnClass', 'Reproduce', 'Figure')

# Hyperparameters and configuration
Run_scanorama_batch_correction = False  # Whether to apply batch correction
NHIDDEN = [100]  # Number of hidden units in the neural network
MAX_ITER = 30  # Maximum iterations for the process
MEMORY_SAVING_MODE = True  # Toggle memory-saving mode
