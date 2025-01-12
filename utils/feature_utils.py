import torch
from rdkit import Chem

# Function to generate the feature matrix (61) 
def generate_feature_matrix(mol):

    atom_types = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Se', 'Si']
    feature_matrix = []

    # electronegativity = {
    #     'As': 2.18, 'B': 2.04, 'Br': 2.96, 'C': 2.55, 'Cl': 3.16, 'F': 3.98, 'I': 2.66,
    #     'N': 3.04, 'O': 3.44, 'P': 2.19, 'S': 2.58, 'Se': 2.55, 'Si': 1.90
    # }
    # atomic_radii = {
    #     'As': 1.85, 'B': 0.87, 'Br': 1.14, 'C': 0.77, 'Cl': 0.99, 'F': 0.64, 'I': 1.33,
    #     'N': 0.75, 'O': 0.73, 'P': 1.00, 'S': 1.04, 'Se': 1.17, 'Si': 1.11
    # }
    valence_electrons = {
        'As': 5, 'B': 3, 'Br': 7, 'C': 4, 'Cl': 7, 'F': 7, 'I': 7,
        'N': 5, 'O': 6, 'P': 5, 'S': 6, 'Se': 6, 'Si': 4
    }

    for atom in mol.GetAtoms():
        features = []

        # Atom type (13)
        atom_symbol = atom.GetSymbol()
        features.extend([1 if atom_symbol == t else 0 for t in atom_types])

        # Number of implicit hydrogens (max: 4)
        num_hydrogens = atom.GetTotalNumHs()
        features.extend([1 if num_hydrogens == i else 0 for i in range(5)])

        # Aromaticity (1)
        features.append(1 if atom.GetIsAromatic() else 0)

        # Hybridization (3)
        hybridization = atom.GetHybridization()
        features.extend([
            1 if hybridization == Chem.rdchem.HybridizationType.SP else 0,
            1 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0,
            1 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0
        ])

        # Formal charge (considered -3 to +3)
        formal_charge = atom.GetFormalCharge()
        features.extend([1 if formal_charge == i else 0 for i in range(-3, 4)])

        # Atomic mass (numerical, scaled)
        # features.append(atom.GetMass() / 100.0)  # Example scaling

        # Electronegativity (numerical, scaled)
        # features.append(electronegativity.get(atom_symbol, 0) / 4.0)

        # Atomic radius (numerical, scaled)
        # features.append(atomic_radii.get(atom_symbol, 0) / 2.0)

        # Valence electrons (numerical, one-hot encoding for 1-8)
        valence = valence_electrons.get(atom_symbol, 0)
        features.extend([1 if valence == i else 0 for i in range(1, 9)])

        # Isotope information (binary)
        # features.append(1 if atom.GetIsotope() else 0)

        # Atom degree (one-hot encoding for 0-4)
        degree = atom.GetDegree()
        features.extend([1 if degree == i else 0 for i in range(5)])

        # Implicit valence (one-hot encoding for 0-8)
        implicit_valence = atom.GetImplicitValence()
        features.extend([1 if implicit_valence == i else 0 for i in range(9)])

        # Total valence (numerical, scaled)
        # features.append(atom.GetTotalValence() / 8.0)

        # Chirality (binary)
        features.append(1 if atom.HasProp('_CIPCode') else 0)

        # Is heteroatom (binary)
        features.append(1 if atom.GetAtomicNum() not in [1, 6] else 0)

        # Is in ring (binary)
        features.append(1 if atom.IsInRing() else 0)

        # Total number of bonds (one-hot encoding for 1-6)
        num_bonds = len(atom.GetBonds())
        features.extend([1 if num_bonds == i else 0 for i in range(1, 7)])

        # Is terminal atom (binary)
        features.append(1 if atom.GetDegree() == 1 else 0)

        feature_matrix.append(features)

    return torch.tensor(feature_matrix, dtype=torch.float32)
