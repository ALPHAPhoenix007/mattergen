# ⚗️ MatterGen: AI-Powered Material Property Predictor

**A machine learning system for predicting material properties from molecular structure**

Built for hackathons | Clean code | Jury-ready | Production-style architecture

---

## 🎯 Project Overview

MatterGen is an AI-powered web application that predicts material properties (band gap, stability, melting point) from chemical formulas or SMILES strings using:
- **Machine Learning**: scikit-learn Random Forest models
- **Chemistry**: RDKit molecular processing
- **Similarity Matching**: Tanimoto similarity with Morgan fingerprints
- **Interactive UI**: Streamlit web interface with 3D visualization

### Key Features
✅ Multiple input methods (SMILES, formula, examples)  
✅ 4 property predictions per molecule  
✅ Similarity matching with known compounds  
✅ Interactive 3D molecular visualization  
✅ Model explainability & feature importance  
✅ Professional scientific UI (no chatbot vibes)

---

## 🏗️ Architecture

```
MatterGen/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── backend/                    # Core ML & chemistry logic
│   ├── chemistry.py           # RDKit molecule processing
│   ├── features.py            # Feature extraction (fingerprints, descriptors)
│   ├── ml_models.py           # ML prediction pipeline
│   └── similarity.py          # Similarity matching engine
│
├── data/                       # Datasets
│   ├── materials_dataset.csv  # Training data (50 compounds)
│   └── sample_inputs.csv      # Example molecules
│
├── models/                     # Saved ML models (generated on first run)
│   └── trained_model.pkl
│
└── utils/                      # Helper functions
    ├── config.py              # Configuration & constants
    └── visualizer.py          # 3D visualization (Py3Dmol)
```

---

## 🚀 Quick Start

### Installation

1. **Clone repository**
```bash
git clone <your-repo>
cd MatterGen
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run application**
```bash
streamlit run app.py
```

4. **Open browser**
Navigate to `http://localhost:8501`

---

## 🧪 How It Works

### 1. Input Processing
- User enters SMILES string (e.g., `c1ccccc1` for benzene) or molecular formula
- RDKit validates and parses the molecule
- System extracts structural features

### 2. Feature Extraction
- **Morgan Fingerprints** (ECFP4): 2048-bit binary vectors for similarity
- **Molecular Descriptors**: 8 key properties (MolWt, LogP, TPSA, etc.)
- Features are normalized using training statistics

### 3. ML Prediction
- **Random Forest Regressors** (one per property)
- Trained on 50 diverse organic/inorganic compounds
- 5-fold cross-validation during training
- Outputs: band gap, formation energy, stability score, melting point

### 4. Similarity Matching
- Compares query molecule to database using Tanimoto similarity
- Returns top-5 most similar known compounds
- Useful for validating predictions against known materials

### 5. Visualization
- Interactive 3D molecular structure using Py3Dmol
- Rotatable, zoomable, multiple rendering styles
- Embedded directly in Streamlit

---

## 📊 Predicted Properties

| Property | Unit | Description |
|----------|------|-------------|
| **Band Gap** | eV | Electronic band gap (semiconductors) |
| **Formation Energy** | eV/atom | Thermodynamic stability |
| **Stability Score** | 0-1 | Chemical stability metric |
| **Melting Point** | K | Phase transition temperature |

---

## 🎨 UI Features

### Professional Scientific Design
- Clean color palette (deep blue, green, white)
- Card-based layout for metrics
- Tabbed interface for organization
- No flashy animations or AI chatbot aesthetics

### Four Main Tabs
1. **📊 Predictions**: Property predictions + molecular info
2. **🔍 Similarity Analysis**: Similar compounds + comparison
3. **🧬 3D Structure**: Interactive molecular visualization
4. **📈 Model Insights**: Feature importance + dataset stats

---

## 🔬 Technical Details

### Machine Learning
- **Model**: Random Forest Regressor (100 trees, max_depth=10)
- **Features**: 8 RDKit molecular descriptors
- **Training**: 80/20 train-test split, 5-fold CV
- **Metrics**: MAE, RMSE, R² score

### Chemistry Engine
- **Library**: RDKit 2023.9.4
- **Fingerprints**: Morgan (circular, radius=2)
- **3D Generation**: MMFF force field optimization
- **Validation**: Molecule sanitization + structural checks

### Similarity Matching
- **Algorithm**: Tanimoto coefficient on Morgan fingerprints
- **Threshold**: Minimum 0.0 (configurable)
- **Results**: Top-5 most similar compounds

---

## 📁 Dataset

**Training Data**: `data/materials_dataset.csv`
- **Size**: 50 compounds
- **Diversity**: Organic (benzene, alcohols) + inorganic (water, salts)
- **Properties**: Band gap, formation energy, stability, melting point
- **Source**: Synthetic data for hackathon (based on realistic values)

**Note**: In production, this would be replaced with databases like:
- Materials Project API
- PubChem
- Cambridge Structural Database

---

## 🎓 Jury Presentation Tips

### What Makes This Project Strong
1. **Full-stack implementation** (backend ML + frontend UI)
2. **Production-quality code** (modular, documented, testable)
3. **Real chemistry** (RDKit is industry-standard)
4. **Explainable AI** (feature importance, similarity matching)
5. **Realistic scope** (hackathon-appropriate, not overengineered)

### Demo Flow
1. Show the UI (professional, clean)
2. Input a simple molecule (benzene)
3. Explain the prediction pipeline
4. Show similar compounds
5. Display 3D structure
6. Show feature importance graph
7. Discuss future improvements

### Key Technical Terms
- **ECFP (Extended Connectivity Fingerprints)**: Industry-standard molecular representation
- **Tanimoto Similarity**: Standard metric for molecular similarity
- **Random Forest**: Interpretable, robust ML algorithm
- **RDKit**: Leading open-source cheminformatics toolkit

---

## 🔮 Future Improvements

### Short-term (Post-Hackathon)
- [ ] Add more training data (expand to 1000+ compounds)
- [ ] Implement ensemble models (RF + Gradient Boosting)
- [ ] Add uncertainty quantification (prediction intervals)
- [ ] Export predictions to PDF report

### Long-term (Production)
- [ ] Connect to Materials Project API
- [ ] Add reaction prediction
- [ ] Multi-objective optimization
- [ ] Deployment on cloud (AWS/GCP)
- [ ] User authentication & saved sessions

---

## 🛠️ Development

### Adding New Properties
1. Add property to `utils/config.py` PROPERTIES list
2. Add column to `data/materials_dataset.csv`
3. Retrain models (automatic on app restart)
4. Update UI in `app.py` (add to property_info dict)

### Testing
```bash
# Test molecule parsing
python -c "from backend.chemistry import smiles_to_mol; print(smiles_to_mol('CCO'))"

# Test feature extraction
python -c "from backend.features import extract_features_from_smiles; print(extract_features_from_smiles('c1ccccc1'))"
```

---

## 🤝 Contributing

This is a hackathon project, but contributions welcome!

1. Fork the repo
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - feel free to use for learning/hackathons

---

## 👨‍💻 Author

Built with ☕ and 🧪 for [Your Hackathon Name]

**Tech Stack**: Python | Streamlit | RDKit | scikit-learn | Plotly

**Questions?** Open an issue or reach out!

---

## 🙏 Acknowledgments

- **RDKit**: Chemistry toolkit
- **Streamlit**: Rapid UI framework
- **Materials Project**: Inspiration for properties
- **scikit-learn**: ML library
- **Py3Dmol**: 3D visualization

## App URL

https://mattergen.streamlit.app/

---

**⭐ Star this repo if you found it useful!**
