"""
Module to augment catalyst data with chemical properties using RDKit and PySCF.
"""
import os
import pandas as pd
import numpy as np
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    print("Rdkit not installed. Molecule features will not be available.")
    Chem = None
    Descriptors = None


def calculate_chem_props():
    """
    Augments the blended dataset with molecular weight, LogP, and synthetic
    electronic properties (using PySCF or fallback).
    """
    print("Augmenting data with Catalyst Properties...")

    # Load Blended Data
    df_path = os.path.join(os.path.dirname(__file__), '../data/blended_v1.csv')
    if not os.path.exists(df_path):
        print(f"Error: {df_path} not found. Run fetch_nrel.py first.")
        return

    df = pd.read_csv(df_path)

    # 1. RDKit: Morphology / Molecular Props
    # Simulate different catalysts being used in different 'runs' or rows
    # Let's say we have 3 catalyst types mixed in the data
    catalysts = {
        'IrO2': '[Ir](=O)=O',
        'Pt/C': '[Pt]',
        'RuO2': '[Ru](=O)=O'
    }

    # Assign random catalyst to each row (simulation)
    # In real life, this would come from the experimental log
    catalyst_names = list(catalysts.keys())
    df['catalyst_type'] = np.random.choice(catalyst_names, len(df))

    # Compute RDKit descriptors
    # Cache them
    props_map = {}
    for name, smiles in catalysts.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            # Proxy for hydrophobicity/surface interaction
            logp = Descriptors.MolLogP(mol)
        else:
            mw, logp = 0, 0
        props_map[name] = (mw, logp)

    df['mol_weight'] = df['catalyst_type'].map(lambda x: props_map[x][0])
    df['logp'] = df['catalyst_type'].map(lambda x: props_map[x][1])

    # 2. PySCF: Electronic Properties (HER Energy)
    # Real PySCF calculation is heavy. Run small one if possible, or mock.
    try:
        # pylint: disable=import-outside-toplevel
        from pyscf import gto, scf
        print("PySCF found. Running electronic structure calc (simplified)...")
        # Calc energy for basic atoms as features
        # Just doing H2 molecule for demo as "system energy" proxy
        mol_h2 = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        mf = scf.RHF(mol_h2)
        h2_energy = mf.kernel()
        # simplified constant for now, or add noise
        df['her_energy'] = h2_energy

    except ImportError:
        print("PySCF not found. Generating detailed synthetic electronic props...")
        # Synthetic HER Energy based on catalyst type
        # Pt is best (-0.2 eV vs RHE approx), IrO2/RuO2 different.
        # Here we put abstract energy values.
        her_map = {
            'IrO2': -150.2,  # Arbitrary Hartree units or similar
            'Pt/C': -155.5,
            'RuO2': -152.1
        }
        df['her_energy'] = df['catalyst_type'].map(lambda x: her_map[x])
        # Add some experimental variation
        df['her_energy'] += np.random.normal(0, 0.1, len(df))

    # 3. Add 'Surface_Area' and 'Porosity'
    # Drop existing if present to avoid confusion
    if 'Surface_Area' in df.columns:
        df = df.drop(columns=['Surface_Area'])
    if 'Porosity' in df.columns:
        df = df.drop(columns=['Porosity'])

    # Randomly correlated with efficiency
    df['Surface_Area'] = np.random.normal(50, 10, len(df))  # m2/g
    df['Surface_Area'] = df['Surface_Area'].abs()  # Ensure positive

    # Porosity (0 to 1)
    df['Porosity'] = np.random.uniform(0.3, 0.7, len(df))

    # Augment efficiency based on these props (make it physics-informed)
    # Higher surface area -> Higher Efficiency
    # Pt -> Higher Efficiency
    # Handle NaNs in base efficiency
    base_eff = df['efficiency'].fillna(df['efficiency'].mean())
    boost = (df['Surface_Area'] - 50) * 0.1
    catalyst_boost = df['catalyst_type'].map(
        {'Pt/C': 2, 'IrO2': 1, 'RuO2': 0.5})

    df['efficiency_augmented'] = base_eff + boost + catalyst_boost

    # Fill remaining NaNs in critical columns used by main_digital_twin
    cols_to_fill = ['voltage', 'current', 'temperature',
                    'pressure', 'degradation', 'efficiency']
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear').fillna(
                method='bfill').fillna(method='ffill')

    # Save Augmented
    out_path = os.path.join(os.path.dirname(__file__),
                            '../data/blended_props.csv')
    df.to_csv(out_path, index=False)
    print(
        f"Augmented data saved to {out_path} with {len(df.columns)} columns.")

    # Verification
    if 'Surface_Area' in df.columns:
        print("Correlation Verification:")
        print(df[['efficiency_augmented', 'Surface_Area', 'mol_weight']].corr()[
              'efficiency_augmented'])


if __name__ == "__main__":
    calculate_chem_props()
