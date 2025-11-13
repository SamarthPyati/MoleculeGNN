import streamlit as st 
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from PIL import Image

from models.gcn import SimpleMoleculeGCN
from models.gin import AdvancedMoleculeGNN
from core.predictor import ModelPredictor
from core.utils import count_parameters
from config.config import ModelConfig, get_device_for_torch

# Page configuration
st.set_page_config(
    page_title="MoleculeGNN - Property Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .molecule-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'predictor' not in st.session_state:
    st.session_state.predictor = None

def detect_model_type(state_dict: dict) -> str:
    """Detect model type from state_dict keys"""
    keys = list(state_dict.keys())
    
    if any('edge_encoder' in key for key in keys) or any('conv' in key and 'nn' in key for key in keys):
        return "gin"
    elif any('conv' in key and 'lin' in key for key in keys) or any('bn' in key for key in keys):
        return "gcn"
    else:
        return "gin"  # Default

def load_pretrained_model(model_path: str, model_type: Optional[str] = None) -> tuple[Optional[ModelPredictor], Optional[str]]:
    """Load pretrained model"""
    try:
        device = get_device_for_torch()
        config = ModelConfig()
        
        # Load state_dict to detect model type
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle both checkpoint format and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            saved_config = checkpoint.get('config', None)
            if saved_config:
                config = saved_config
        else:
            state_dict = checkpoint
        
        # Auto-detect model type if not specified
        detected_type = detect_model_type(state_dict)
        actual_model_type = (model_type.lower() if model_type else detected_type)
        
        # Try loading with detected/selected type
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
                continue
        
        if model is None:
            return None, None
        
        model = model.to(device)
        model.eval()
        predictor = ModelPredictor(model, device=device)
        return predictor, actual_model_type
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def draw_molecule(smiles: str, size=(400, 400)) -> Optional[Image.Image]:
    """Draw molecule from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None

def get_molecular_properties(smiles: str) -> Optional[dict]:
    """Calculate molecular properties"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'Molecular Weight': f"{Descriptors.MolWt(mol):.2f}",  # type: ignore
            'LogP': f"{Descriptors.MolLogP(mol):.2f}",  # type: ignore
            'H-Bond Donors': Descriptors.NumHDonors(mol),  # type: ignore
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),  # type: ignore
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),  # type: ignore
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),  # type: ignore
            'TPSA': f"{Descriptors.TPSA(mol):.2f}",  # type: ignore
        }
    except Exception:
        return None

def create_similarity_chart(smiles_list: list, predictions: list):
    """Create a chart showing prediction similarity"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(predictions))),
        y=predictions,
        mode='lines+markers',
        name='Predictions',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Prediction History',
        xaxis_title='Molecule Index',
        yaxis_title='Predicted Property',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Header
st.markdown('<h1 class="main-header">🧬 MoleculeGNN Property Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict molecular properties using Graph Neural Networks")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Settings")
    model_type = st.selectbox(
        "Model Architecture",
        ["Auto-detect", "GCN", "GIN"],
        help="Choose between Graph Convolutional Network or Graph Isomorphism Network"
    )
    
    # Model path input
    st.subheader("📁 Load Model")
    model_path = st.text_input(
        "Model Path",
        value="notebooks/best_model.pt",
        help="Path to the pretrained model checkpoint"
    )
    
    # File uploader for model
    model_file = st.file_uploader(
        "Or Upload Model (.pt)",
        type=['pt'],
        help="Upload a pretrained model checkpoint"
    )
    
    if model_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp_model.pt")
        with open(temp_path, "wb") as f:
            f.write(model_file.getvalue())
        model_path = str(temp_path)
    
    # Load model button
    load_model_btn = st.button("🔄 Load Model", type="primary")
    
    if load_model_btn:
        with st.spinner("Loading model..."):
            selected_type = None if model_type == "Auto-detect" else model_type.lower()
            predictor, detected_type = load_pretrained_model(model_path, selected_type)
            
            if predictor and detected_type:
                st.session_state.model_loaded = True
                st.session_state.predictor = predictor
                st.session_state.detected_model_type = detected_type.upper()
                st.success(f"✅ Model loaded successfully! (Detected: {detected_type.upper()})")
                
                # Show model info
                try:
                    num_params = count_parameters(predictor.model)
                    st.info(f"📊 Model Parameters: {num_params:,}")
                except Exception:
                    pass
            else:
                st.error("❌ Failed to load model. Please check the model path.")
                st.session_state.model_loaded = False
    
    # Show current model status
    if st.session_state.model_loaded:
        detected_type = st.session_state.get('detected_model_type', 'Unknown')
        st.info(f"ℹ️ Model loaded: {detected_type}")
    
    st.markdown("---")
    
    # Task type
    task_type = st.radio(
        "Prediction Task",
        ["Solubility", "Toxicity", "Custom"],
        help="Select the type of property to predict"
    )
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.info("""
    This application uses Graph Neural Networks to predict molecular properties.
    
    **Features:**
    - Single molecule prediction
    - Batch prediction from CSV
    - Molecular property calculation
    - Interactive visualizations
    """)
    
    # Stats
    if st.session_state.model_loaded:
        st.markdown("---")
        st.subheader("📊 Session Stats")
        st.metric("Predictions Made", len(st.session_state.predictions))

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Single Prediction", "📊 Batch Prediction", "📈 Analysis", "ℹ️ Documentation"])

with tab1:
    st.header("Single Molecule Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Molecule")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["SMILES String", "Draw Structure (Coming Soon)"],
            horizontal=True
        )
        
        if input_method == "SMILES String":
            # Example molecules
            st.write("**Example molecules:**")
            examples = {
                "Ethanol": "CCO",
                "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            }
            
            selected_example = st.selectbox("Select an example:", ["Custom"] + list(examples.keys()))
            
            if selected_example != "Custom":
                smiles_input = examples[selected_example]
            else:
                smiles_input = st.text_input(
                    "Enter SMILES String",
                    value="CCO",
                    help="Enter the SMILES representation of your molecule"
                )
            
            # Draw molecule
            if smiles_input:
                img = draw_molecule(smiles_input)
                if img:
                    st.image(img, caption="Molecular Structure", use_container_width=True)
                else:
                    st.error("❌ Invalid SMILES string")
            
            # Molecular properties
            if smiles_input:
                props = get_molecular_properties(smiles_input)
                if props:
                    st.subheader("Molecular Properties")
                    prop_col1, prop_col2 = st.columns(2)
                    
                    with prop_col1:
                        for key in list(props.keys())[:4]:
                            st.metric(key, props[key])
                    
                    with prop_col2:
                        for key in list(props.keys())[4:]:
                            st.metric(key, props[key])
    
    with col2:
        st.subheader("Prediction Results")
        
        if st.button("🚀 Predict", type="primary", disabled=not st.session_state.model_loaded):
            if not st.session_state.model_loaded or st.session_state.predictor is None:
                st.warning("⚠️ Please load a model first!")
            else:
                with st.spinner("Predicting..."):
                    try:
                        return_prob = (task_type == "Toxicity")
                        predictor = st.session_state.predictor
                        prediction = predictor.predict_smiles(
                            smiles_input,
                            return_probability=return_prob
                        )
                        
                        if prediction is not None:
                            # Store prediction
                            st.session_state.predictions.append({
                                'smiles': smiles_input,
                                'prediction': prediction
                            })
                            
                            # Display result
                            st.markdown('<div class="molecule-card">', unsafe_allow_html=True)
                            st.markdown(f"### Prediction: {prediction:.4f}")
                            
                            # Interpretation
                            if task_type == "Toxicity":
                                if prediction > 0.5:
                                    st.error("⚠️ High Toxicity Risk")
                                else:
                                    st.success("✅ Low Toxicity Risk")
                            elif task_type == "Solubility":
                                st.info(f"💧 Log Solubility: {prediction:.4f} mol/L")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Confidence visualization
                            st.subheader("Confidence Distribution")
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prediction * 100 if task_type == "Toxicity" else prediction,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Confidence Score"},
                                gauge={
                                    'axis': {'range': [None, 100] if task_type == "Toxicity" else None},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 33], 'color': "lightgray"},
                                        {'range': [33, 66], 'color': "gray"},
                                        {'range': [66, 100], 'color': "darkgray"}
                                    ],
                                }
                            ))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            st.error("❌ Prediction failed. Please check your SMILES string.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
        else:
            if not st.session_state.model_loaded:
                st.info("👈 Please load a model first using the sidebar")

with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with SMILES strings to predict multiple molecules at once.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV should have a column containing SMILES strings"
    )
    
    if uploaded_file is not None:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.write(f"**Loaded {len(df)} molecules**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Column selection
        smiles_column = st.selectbox(
            "Select SMILES column",
            df.columns.tolist()
        )
        
        if st.button("🚀 Predict Batch", type="primary", disabled=not st.session_state.model_loaded):
            if not st.session_state.model_loaded or st.session_state.predictor is None:
                st.warning("⚠️ Please load a model first!")
            else:
                with st.spinner(f"Predicting {len(df)} molecules..."):
                    try:
                        smiles_list = df[smiles_column].tolist()
                        return_prob = (task_type == "Toxicity")
                        predictor = st.session_state.predictor
                        predictions = predictor.predict_batch(
                            smiles_list,
                            batch_size=32,
                            return_probability=return_prob
                        )
                        
                        # Add predictions to dataframe
                        df['Prediction'] = predictions
                        
                        # Display results
                        st.success(f"✅ Predicted {len(df)} molecules successfully!")
                        
                        # Statistics
                        valid_preds = predictions[~np.isnan(predictions)]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Molecules", len(df))
                        with col2:
                            st.metric("Mean Prediction", f"{np.nanmean(predictions):.4f}")
                        with col3:
                            st.metric("Std Deviation", f"{np.nanstd(predictions):.4f}")
                        with col4:
                            st.metric("Valid Predictions", f"{len(valid_preds)}")
                        
                        # Results table
                        st.subheader("Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        if len(valid_preds) > 0:
                            st.subheader("Distribution of Predictions")
                            fig = px.histogram(
                                df.dropna(subset=['Prediction']), 
                                x='Prediction',
                                nbins=50,
                                title="Prediction Distribution",
                                labels={'Prediction': 'Predicted Value'},
                                color_discrete_sequence=['#667eea']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
    else:
        st.info("👆 Upload a CSV file to start batch prediction")

with tab3:
    st.header("Analysis & Insights")
    
    if len(st.session_state.predictions) > 0:
        # Create dataframe from predictions
        pred_df = pd.DataFrame(st.session_state.predictions)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction History")
            fig = create_similarity_chart(
                pred_df['smiles'].tolist(),
                pred_df['prediction'].tolist()
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution Analysis")
            fig = px.box(
                pred_df,
                y='prediction',
                title="Prediction Distribution",
                labels={'prediction': 'Predicted Value'},
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("All Predictions")
        st.dataframe(pred_df, use_container_width=True)
        
        # Download all predictions
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions",
            data=csv,
            file_name="all_predictions.csv",
            mime="text/csv"
        )
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.predictions = []
            st.rerun()
    else:
        st.info("👆 Make some predictions first to see analysis!")

with tab4:
    st.header("Documentation")
    
    st.markdown("""
    ## 🧬 MoleculeGNN - Graph Neural Network for Molecular Property Prediction
    
    ### Overview
    This application uses Graph Neural Networks (GNNs) to predict molecular properties. 
    GNNs are particularly well-suited for molecular data because they can naturally represent 
    the graph structure of molecules (atoms as nodes, bonds as edges).
    
    ### Features
    
    #### 🔬 Single Prediction
    - Input molecules using SMILES notation
    - View 2D molecular structure
    - Calculate standard molecular properties
    - Get instant predictions with confidence scores
    
    #### 📊 Batch Prediction
    - Upload CSV files with multiple molecules
    - Process hundreds of molecules at once
    - Download results for further analysis
    - Visualize prediction distributions
    
    #### 📈 Analysis
    - Track prediction history
    - Analyze trends and distributions
    - Export data for external analysis
    
    ### Supported Properties
    
    The model can predict various molecular properties depending on how it was trained:
    
    - **Solubility**: Water solubility in mol/L
    - **Toxicity**: Binary toxicity classification
    - **Lipophilicity**: Octanol-water partition coefficient
    - **And more...**
    
    ### How to Use
    
    1. **Load a Model**: Upload a pretrained `.pt` model file in the sidebar
    2. **Enter Molecule**: Use SMILES notation or select an example
    3. **Predict**: Click the predict button to get results
    4. **Analyze**: View molecular properties and prediction confidence
    
    ### About SMILES
    
    SMILES (Simplified Molecular Input Line Entry System) is a notation for representing 
    chemical structures using ASCII strings. Examples:
    
    - Water: `O`
    - Ethanol: `CCO`
    - Benzene: `c1ccccc1`
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    
    ### Model Architecture
    
    The application supports two GNN architectures:
    
    - **GCN (Graph Convolutional Network)**: Standard message passing
    - **GIN (Graph Isomorphism Network)**: More expressive architecture
    
    ### Technical Details
    
    - **Framework**: PyTorch + PyTorch Geometric
    - **Chemistry**: RDKit for molecular processing
    - **UI**: Streamlit for interactive interface
    
    ### Contact & Support
    
    For questions or issues, please visit the project repository.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2025 MoleculeGNN Project, Samarth Pyati</p>
</div>
""", unsafe_allow_html=True)
