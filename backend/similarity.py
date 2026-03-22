"""
Similarity matching engine
Finds similar molecules in database using fingerprint similarity
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from backend.chemistry import smiles_to_mol
from backend.features import generate_morgan_fingerprint


class SimilarityEngine:
    """
    Engine for finding similar molecules based on fingerprint similarity
    """
    
    def __init__(self):
        self.database_fps = []
        self.database_smiles = []
        self.database_info = None
    
    def load_database(self, df: pd.DataFrame, smiles_col: str = 'smiles'):
        """
        Load molecule database for similarity searching
        
        Args:
            df: DataFrame with molecules
            smiles_col: Column name containing SMILES
        """
        self.database_info = df.copy()
        self.database_smiles = df[smiles_col].tolist()
        
        # Generate fingerprints for all molecules
        self.database_fps = []
        for smiles in self.database_smiles:
            mol = smiles_to_mol(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                self.database_fps.append(fp)
            else:
                self.database_fps.append(None)
    
    def find_similar(self, 
                    query_smiles: str, 
                    n_results: int = 5,
                    min_similarity: float = 0.0) -> List[Tuple[int, float, dict]]:
        """
        Find most similar molecules to query
        
        Args:
            query_smiles: SMILES string of query molecule
            n_results: Number of similar molecules to return
            min_similarity: Minimum Tanimoto similarity threshold
        
        Returns:
            List of tuples: (index, similarity_score, molecule_info)
        """
        # Generate fingerprint for query
        query_mol = smiles_to_mol(query_smiles)
        if query_mol is None:
            return []
        
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2, nBits=2048)
        
        # Calculate similarities
        similarities = []
        for idx, db_fp in enumerate(self.database_fps):
            if db_fp is not None:
                # Tanimoto similarity
                similarity = DataStructs.TanimotoSimilarity(query_fp, db_fp)
                if similarity >= min_similarity:
                    similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N results
        results = []
        for idx, sim in similarities[:n_results]:
            mol_info = self.database_info.iloc[idx].to_dict()
            results.append((idx, sim, mol_info))
        
        return results
    
    def calculate_diversity_score(self, smiles_list: List[str]) -> float:
        """
        Calculate diversity score for a set of molecules
        Higher score = more diverse
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            Diversity score (0-1)
        """
        if len(smiles_list) < 2:
            return 0.0
        
        # Generate fingerprints
        fps = []
        for smiles in smiles_list:
            mol = smiles_to_mol(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
        
        if len(fps) < 2:
            return 0.0
        
        # Calculate average pairwise dissimilarity
        total_dissimilarity = 0
        count = 0
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dissimilarity = 1 - similarity
                total_dissimilarity += dissimilarity
                count += 1
        
        diversity = total_dissimilarity / count if count > 0 else 0
        return diversity


def calculate_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]:
    """
    Calculate Tanimoto similarity between two molecules
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
    
    Returns:
        Similarity score (0-1) or None if error
    """
    mol1 = smiles_to_mol(smiles1)
    mol2 = smiles_to_mol(smiles2)
    
    if mol1 is None or mol2 is None:
        return None
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity
