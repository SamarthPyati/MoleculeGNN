import streamlit as st
import torch
from pathlib import Path
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

from models.gcn import SimpleMoleculeGCN
from models.gin import AdvancedMoleculeGNN
from core.predictor import ModelPredictor
from config.config import ModelConfig, get_device_for_torch

# Page configuration
st.set_page_config(
    page_title="MoleculeGNN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .molecule-container {
        text-align: center;
        padding: 1rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def detect_model_type(state_dict: dict) -> str:
    """Detect model type from state_dict keys"""
    # GIN models have 'edge_encoder' and 'conv*.nn.*' keys
    # GCN models have 'conv*.lin.weight' and 'bn*' keys
    keys = list(state_dict.keys())
    
    if any('edge_encoder' in key for key in keys) or any('conv' in key and 'nn' in key for key in keys):
        return "gin"
    elif any('conv' in key and 'lin' in key for key in keys) or any('bn' in key for key in keys):
        return "gcn"
    else:
        # Default to GIN if we can't determine (more common)
        return "gin"

@st.cache_resource
def load_model(model_path: str, model_type: Optional[str] = None) -> tuple[Optional[ModelPredictor], Optional[str]]:
    """Load a trained model from disk
    
    Returns:
        Tuple of (predictor, detected_model_type) or (None, None) on error
    """
    try:
        if not Path(model_path).exists():
            return None, None
        
        device = get_device_for_torch()
        config = ModelConfig()
        
        # Load state_dict to detect model type
        state_dict = torch.load(model_path, map_location=device)
        
        # Auto-detect model type if not specified or if specified type fails
        detected_type = detect_model_type(state_dict)
        actual_model_type = model_type.lower() if model_type else detected_type
        
        # Try loading with detected/selected type first
        model = None
        for attempt_type in [actual_model_type, detected_type]:
            try:
                if attempt_type == "gcn":
                    model = SimpleMoleculeGCN(
                        num_node_features=config.num_node_features,
                        hidden_dim=config.hidden_dim,
                        num_classes=config.num_classes,
                        dropout=config.dropout
                    )
                else:  # gin
                    model = AdvancedMoleculeGNN(
                        num_node_features=config.num_node_features,
                        num_edge_features=config.num_edge_features,
                        hidden_dim=config.hidden_dim,
                        num_classes=config.num_classes,
                        dropout=config.dropout
                    )
                
                model.load_state_dict(state_dict, strict=False)
                actual_model_type = attempt_type
                break
            except Exception:
                # If this attempt failed, try the other model type
                continue
        
        if model is None:
            return None, None
        
        predictor = ModelPredictor(model, device=device)
        return predictor, actual_model_type
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def draw_molecule(smiles: str, width: int = 400, height: int = 300) -> Optional[Image.Image]:
    """Draw a molecule from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(width, height))
        return img
    except Exception:
        return None

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def main() -> None:
    # Header
    st.markdown('<h1 class="main-header">🧬 MoleculeGNN</h1>', unsafe_allow_html=True)
    st.markdown("### Predict molecular properties using Graph Neural Networks")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        # Model type selection (optional - will auto-detect if not specified)
        model_type = st.selectbox(
            "Model Architecture (Auto-detected if not specified)",
            ["Auto-detect", "GCN", "GIN"],
            help="Auto-detect: Automatically detect from model file\nGCN: Graph Convolutional Network\nGIN: Graph Isomorphism Network"
        )
        
        # Model path input
        st.subheader("📁 Load Model")
        model_path = st.text_input(
            "Model Path",
            value="notebooks/best_model.pt",
            help="Path to the trained model checkpoint"
        )
        
        # Load model button
        load_model_btn = st.button("🔄 Load Model", type="primary")
        
        # Task type
        task_type = st.selectbox(
            "Task Type",
            ["Regression", "Classification"],
            help="Regression: Continuous values\nClassification: Binary classification"
        )
        
        # Property name
        property_name = st.text_input(
            "Property Name",
            value="Molecular Property",
            help="Name of the property being predicted"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        **MoleculeGNN** uses Graph Neural Networks to predict molecular properties from SMILES strings.
        
        - Convert SMILES to molecular graphs
        - Extract atom and bond features
        - Predict properties using trained GNN models
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Input mode selection
        input_mode = st.radio(
            "Input Mode",
            ["Single Molecule", "Batch Prediction"],
            horizontal=True
        )
        
        if input_mode == "Single Molecule":
            # Single SMILES input
            smiles_input = st.text_input(
                "Enter SMILES String",
                placeholder="e.g., CCO (ethanol)",
                help="Enter a valid SMILES string representing a molecule"
            )
            
            if smiles_input:
                # Validate SMILES
                if validate_smiles(smiles_input):
                    st.success("✓ Valid SMILES string")
                    
                    # Draw molecule
                    st.subheader("🔬 Molecule Visualization")
                    mol_img = draw_molecule(smiles_input)
                    if mol_img:
                        st.image(mol_img, use_container_width=True)
                    else:
                        st.warning("Could not visualize molecule")
                else:
                    st.error("❌ Invalid SMILES string")
                    smiles_input = None
        else:
            # Batch input
            batch_input = st.text_area(
                "Enter SMILES Strings (one per line)",
                placeholder="CCO\nCC(=O)O\nC1=CC=CC=C1",
                height=200,
                help="Enter multiple SMILES strings, one per line"
            )
            
            if batch_input:
                smiles_list = [s.strip() for s in batch_input.split("\n") if s.strip()]
                valid_smiles = [s for s in smiles_list if validate_smiles(s)]
                invalid_count = len(smiles_list) - len(valid_smiles)
                
                if invalid_count > 0:
                    st.warning(f"⚠️ {invalid_count} invalid SMILES string(s) found")
                
                st.info(f"📊 {len(valid_smiles)} valid molecule(s) ready for prediction")
                smiles_input = valid_smiles if valid_smiles else None
            else:
                smiles_input = None
    
    with col2:
        st.header("🎯 Prediction")
        
        # Load model if button clicked
        predictor = None
        detected_model_type = None
        
        if load_model_btn or st.session_state.get('model_loaded', False):
            with st.spinner("Loading model..."):
                # Use None for model_type if auto-detect is selected
                selected_type = None if model_type == "Auto-detect" else model_type.lower()
                predictor, detected_model_type = load_model(model_path, selected_type)
                
                if predictor and detected_model_type:
                    st.session_state['model_loaded'] = True
                    st.session_state['predictor'] = predictor
                    st.session_state['detected_model_type'] = detected_model_type.upper()
                    st.success(f"✅ Model loaded successfully! (Detected: {detected_model_type.upper()})")
                else:
                    st.error("❌ Failed to load model. Please check the model path.")
                    st.session_state['model_loaded'] = False
        elif st.session_state.get('predictor'):
            predictor = st.session_state['predictor']
            detected_model_type = st.session_state.get('detected_model_type', 'Unknown')
            st.info(f"ℹ️ Using previously loaded model ({detected_model_type})")
        
        # Prediction button
        if predictor and smiles_input:
            if st.button("🚀 Predict", type="primary", use_container_width=True):
                with st.spinner("Making prediction..."):
                    try:
                        if isinstance(smiles_input, str):
                            # Single prediction
                            prediction = predictor.predict_smiles(
                                smiles_input,
                                return_probability=(task_type == "Classification")
                            )
                            
                            if prediction is not None:
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                st.metric(
                                    label=property_name,
                                    value=f"{prediction:.4f}"
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Additional info
                                with st.expander("📈 Prediction Details"):
                                    actual_model = detected_model_type if detected_model_type else (model_type if model_type != "Auto-detect" else "Unknown")
                                    st.write(f"**SMILES:** `{smiles_input}`")
                                    st.write(f"**Model:** {actual_model}")
                                    st.write(f"**Task:** {task_type}")
                                    st.write(f"**Prediction:** {prediction:.6f}")
                            else:
                                st.error("Failed to make prediction. Invalid molecule.")
                        else:
                            # Batch prediction
                            predictions = predictor.predict_batch(
                                smiles_input,
                                return_probability=(task_type == "Classification")
                            )
                            
                            # Display results
                            results_data = {
                                "SMILES": smiles_input,
                                "Prediction": predictions
                            }
                            
                            import pandas as pd
                            results_df = pd.DataFrame(results_data)
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Statistics
                            valid_preds = predictions[~pd.isna(predictions)]
                            if len(valid_preds) > 0:
                                st.metric(
                                    label="Average Prediction",
                                    value=f"{valid_preds.mean():.4f}"
                                )
                                
                                col_stat1, col_stat2 = st.columns(2)
                                with col_stat1:
                                    st.metric("Min", f"{valid_preds.min():.4f}")
                                with col_stat2:
                                    st.metric("Max", f"{valid_preds.max():.4f}")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        elif not predictor:
            st.info("👈 Please load a model first using the sidebar")
        elif not smiles_input:
            st.info("👈 Please enter a SMILES string to make predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>MoleculeGNN - Predicting molecular properties using Graph Neural Networks</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()