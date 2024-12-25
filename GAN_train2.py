# %%
from rdkit import Chem
from model import MolGen
import numpy as np
import pandas as pd
import torch

# load data
data = []
with open('Smiles_list2.csv', "r") as f:
    for line in f.readlines()[1:]:
        data.append(line.split(",")[0])

# create model
gan_mol = MolGen(data, hidden_dim=64, lr=1e-3, device="cuda")


# %%
# create dataloader
loader = gan_mol.create_dataloader(data, batch_size=512, shuffle=True, num_workers=10)

# train model for 10000 steps and save progress
model_save_path = "gan_mol_model.pth"
log_file = "training_logs.txt"
max_step = 30000
evaluate_every = 200

with open(log_file, "w") as log:
    for step in range(0, max_step, evaluate_every):
        gan_mol.train_n_steps(loader, max_step=evaluate_every, evaluate_every=evaluate_every)
        log.write(f"Step {step + evaluate_every}: Model trained\n")
        log.flush()

        # Save model periodically
        torch.save(gan_mol, model_save_path)
        log.write(f"Step {step + evaluate_every}: Model saved at {model_save_path}\n")

# %%
gan_mol.eval()
print('Model evaluation completed')

# Save generated SMILES molecules
output_file = "generated_smiles.txt"
smiles_list = gan_mol.generate_n(1000)
with open(output_file, "w") as f:
    f.writelines("\n".join(smiles_list))
print(f"Generated SMILES saved to {output_file}")
