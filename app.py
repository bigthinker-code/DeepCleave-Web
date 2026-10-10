import streamlit as st
import os
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepCleave V2.0", page_icon="🧬", layout="centered")

# --- LOAD MODELS (CACHED FOR SPEED) ---
@st.cache_resource
def load_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformers
    prot_tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    prot_mod = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device).eval()
    mol_tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    mol_mod = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR").to(device).eval()
    
    # Classical Models
    xgb = joblib.load("./models/xgb_model.pkl")
    rf = joblib.load("./models/rf_model.pkl")
    meta_judge = joblib.load("./models/meta_judge.pkl")
    p_scaler = joblib.load("./models/p_scaler.pkl")
    m_scaler = joblib.load("./models/m_scaler.pkl")
    
    return device, prot_tok, prot_mod, mol_tok, mol_mod, xgb, rf, meta_judge, p_scaler, m_scaler

device, prot_tok, prot_mod, mol_tok, mol_mod, xgb, rf, meta_judge, p_scaler, m_scaler = load_system()

# --- FEATURE ENGINE ---
def extract_advanced_features(seq, smiles):
    # Deep Learning Intuition
    p_in = prot_tok(seq, return_tensors="pt", truncation=True, max_length=512).to(device)
    m_in = mol_tok(smiles, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        p_emb = prot_mod(**p_in).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        m_emb = mol_mod(**m_in).last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    
    # Advanced Substrate Chemistry
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
        
    m_desc_raw = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol), Lipinski.NumRotatableBonds(mol),
        Fragments.fr_ester(mol), Fragments.fr_amide(mol), Fragments.fr_ether(mol), Fragments.fr_COO(mol),
        Fragments.fr_Al_OH(mol), Fragments.fr_Ar_OH(mol), Fragments.fr_halogen(mol), Fragments.fr_phos_ester(mol)
    ], dtype=np.float32)
    
    # Advanced Protein Biophysics
    L = max(len(seq), 1)
    p_desc_raw = np.array([
        L,
        (seq.count('K') + seq.count('R') + seq.count('H')) / L,
        (seq.count('D') + seq.count('E')) / L,
        (seq.count('F') + seq.count('Y') + seq.count('W')) / L,
        (seq.count('A') + seq.count('V') + seq.count('I') + seq.count('L') + seq.count('M')) / L,
        (seq.count('S') + seq.count('T') + seq.count('N') + seq.count('Q')) / L
    ], dtype=np.float32)
    
    p_desc_scaled = p_scaler.transform(p_desc_raw.reshape(1, -1))
    m_desc_scaled = m_scaler.transform(m_desc_raw.reshape(1, -1))
    
    return np.concatenate((p_emb.reshape(1,-1), p_desc_scaled, m_emb.reshape(1,-1), m_desc_scaled), axis=1)

# --- USER INTERFACE ---
st.title("🧬 DeepCleave V2.0")
st.subheader("Dual-Input Biochemical Reaction Evaluator")
st.markdown("Predicts whether a specific wild-type enzyme will cleave a given small molecule substrate using a Hybrid Stacked Ensemble (ESM-2 + ChemBERTa + XGBoost).")

with st.form("prediction_form"):
    enzyme_seq = st.text_area("Enzyme Sequence (Amino Acids)", height=150, placeholder="e.g., MQIFVKTLTG...")
    substrate_smiles = st.text_input("Substrate Structure (SMILES)", placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O")
    submit_button = st.form_submit_button("Predict Reaction")

if submit_button:
    if not enzyme_seq or not substrate_smiles:
        st.warning("Please provide both an Enzyme Sequence and a Substrate SMILES.")
    else:
        with st.spinner("Analyzing biophysics and molecular structure..."):
            hyb_vec = extract_advanced_features(enzyme_seq, substrate_smiles)
            
            if hyb_vec is None:
                st.error("Invalid SMILES string. RDKit could not parse the molecule.")
            else:
                xgb_vote = float(xgb.predict_proba(hyb_vec)[0][1])
                rf_vote = float(rf.predict_proba(hyb_vec)[0][1])
                final_prob = float(meta_judge.predict_proba(np.array([[xgb_vote, rf_vote]]))[0][1])
                
                st.markdown("---")
                if final_prob > 0.70:
                    st.success(f"**VERDICT: ACTIVE** (Probability: {final_prob:.4f})")
                    st.write("DeepCleave predicts a highly likely reaction.")
                else:
                    st.error(f"**VERDICT: INACTIVE** (Probability: {final_prob:.4f})")
                    st.write("DeepCleave predicts no meaningful reaction.")
                
                st.markdown("### Ensemble Breakdown")
                col1, col2 = st.columns(2)
                col1.metric("XGBoost Confidence", f"{xgb_vote:.4f}")
                col2.metric("Random Forest Confidence", f"{rf_vote:.4f}")