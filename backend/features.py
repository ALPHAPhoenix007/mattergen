"""
Feature extraction module
Generates molecular fingerprints and descriptor vectors for ML models
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from typing import Optional, List
from utils.config import FINGERPRINT_CONFIG, DESCRIPTORS


def generate_morgan_fingerprint(mol: Chem.Mol, 
                                radius: int = FINGERPRINT_CONFIG['radius'],
                                n_bits: int = FINGERPRINT_CONFIG['n_bits']) -> Optional[np.ndarray]:
    """
    Generate Morgan (ECFP) fingerprint for molecule
    
    Args:
        mol: RDKit molecule object
        radius: Fingerprint radius (default: 2 for ECFP4)
        n_bits: Fingerprint length
    
    Returns:
        NumPy array of fingerprint bits or None
    """
    if mol is None:
        return None
    
    try:
        # Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=radius, 
            nBits=n_bits,
            useFeatures=FINGERPRINT_CONFIG['use_features']
        )
        
        # Convert to numpy array
        arr = np.zeros((1,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        
        return arr
    except:
        return None


def calculate_descriptor_vector(mol: Chem.Mol, 
                                descriptor_names: List[str] = DESCRIPTORS) -> Optional[np.ndarray]:
    """
    Calculate molecular descriptors as a feature vector
    
    Args:
        mol: RDKit molecule object
        descriptor_names: List of descriptor names to calculate
    
    Returns:
        NumPy array of descriptor values
    """
    if mol is None:
        return None
    
    descriptor_values = []
    
    for desc_name in descriptor_names:
        try:
            # Get descriptor function from Descriptors module
            desc_func = getattr(Descriptors, desc_name)
            value = desc_func(mol)
            descriptor_values.append(value)
        except:
            # If descriptor fails, use 0
            descriptor_values.append(0.0)
    
    return np.array(descriptor_values)


def generate_combined_features(mol: Chem.Mol) -> Optional[np.ndarray]:
    """
    Generate combined feature vector (fingerprint + descriptors)
    This is the main feature vector used for ML predictions
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        Combined feature vector
    """
    if mol is None:
        return None
    
    # Get fingerprint
    fp = generate_morgan_fingerprint(mol)
    if fp is None:
        return None
    
    # Get descriptors
    descriptors = calculate_descriptor_vector(mol)
    if descriptors is None:
        return None
    
    # Combine features
    # Note: For hackathon, we'll use just descriptors for interpretability
    # Fingerprints will be used for similarity matching
    return descriptors


def extract_features_from_smiles(smiles: str) -> Optional[np.ndarray]:
    """
    End-to-end feature extraction from SMILES string
    
    Args:
        smiles: SMILES representation
    
    Returns:
        Feature vector ready for ML model
    """
    from backend.chemistry import smiles_to_mol
    
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    features = generate_combined_features(mol)
    return features


def get_feature_names() -> List[str]:
    """
    Get names of features in the feature vector
    Used for model interpretation
    
    Returns:
        List of feature names
    """
    return DESCRIPTORS.copy()


def normalize_features(features: np.ndarray, 
                      mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize features using Z-score normalization
    
    Args:
        features: Feature array
        mean: Mean values (if None, calculate from features)
        std: Standard deviation values
    
    Returns:
        Normalized features
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (features - mean) / std
    return normalized
