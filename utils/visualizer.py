"""
3D Molecular visualization using Py3Dmol and RDKit
"""

import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
import streamlit.components.v1 as components


def generate_3d_view(smiles: str, 
                    width: int = 600, 
                    height: int = 400,
                    style: str = 'stick') -> str:
    """
    Generate interactive 3D molecular visualization
    
    Args:
        smiles: SMILES string of molecule
        width: Viewer width in pixels
        height: Viewer height in pixels
        style: Visualization style ('stick', 'sphere', 'cartoon')
    
    Returns:
        HTML string for embedding in Streamlit
    """
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<p>Invalid molecule</p>"
    
    # Add hydrogens for realistic 3D structure
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        return "<p>Could not generate 3D structure</p>"
    
    # Convert to mol block format
    mol_block = Chem.MolToMolBlock(mol)
    
    # Create 3D viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(mol_block, 'mol')
    
    # Set style
    if style == 'stick':
        viewer.setStyle({'stick': {'radius': 0.15}})
    elif style == 'sphere':
        viewer.setStyle({'sphere': {'scale': 0.3}})
    elif style == 'cartoon':
        viewer.setStyle({'cartoon': {'color': 'spectrum'}})
    else:
        viewer.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    
    # Add background and zoom
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    
    # Return HTML
    return viewer._make_html()


def render_3d_molecule(smiles: str, style: str = 'stick'):
    """
    Render 3D molecule in Streamlit
    
    Args:
        smiles: SMILES string
        style: Visualization style
    """
    html = generate_3d_view(smiles, width=800, height=500, style=style)
    components.html(html, width=800, height=500, scrolling=False)


def generate_multiple_conformers(smiles: str, n_conformers: int = 5) -> list:
    """
    Generate multiple 3D conformers for a molecule
    Useful for showing structural flexibility
    
    Args:
        smiles: SMILES string
        n_conformers: Number of conformers to generate
    
    Returns:
        List of conformer energies
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    conformer_ids = AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=n_conformers,
        randomSeed=42
    )
    
    # Optimize and get energies
    energies = []
    for conf_id in conformer_ids:
        try:
            # MMFF optimization
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            ff.Minimize()
            energy = ff.CalcEnergy()
            energies.append(energy)
        except:
            energies.append(None)
    
    return energies
