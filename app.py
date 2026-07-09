import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')

# ===================== Page Basic Configuration =====================
st.set_page_config(page_title="Impact Displacement Prediction of CFST", layout="wide")
st.title("Impact Displacement Prediction System for recycled aggregate concrete-filled steel tube")


# ===================== Model Classes (Identical to training code, DO NOT MODIFY) =====================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class NLPDataProcessor(nn.Module):
    def __init__(self, embed_dim=64, value_hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_hidden_dim = value_hidden_dim
        self.feature_name_vectorizer = CountVectorizer(
            token_pattern=r'(?u)\b\w[\w_]*\b', lowercase=False
        )
        self.name_projection = None
        self.value_encoder = nn.Sequential(
            nn.Linear(1, value_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(value_hidden_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.reg_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, feature_names, feature_values):
        batch_size, num_features = feature_values.shape
        device = feature_values.device
        feature_name_texts = [name for batch in feature_names for name in batch]
        name_vectors = self.feature_name_vectorizer.transform(feature_name_texts).toarray()
        name_vectors = torch.tensor(name_vectors, dtype=torch.float32).to(device)
        name_emb = self.name_projection(name_vectors).reshape(
            batch_size, num_features, self.embed_dim
        )
        values_flat = feature_values.reshape(-1, 1)
        value_emb_flat = self.value_encoder(values_flat)
        value_emb = value_emb_flat.view(batch_size, num_features, self.embed_dim)
        token_emb = name_emb * value_emb
        token_emb = self.norm(token_emb)
        reg_emb = self.reg_embedding.repeat(batch_size, 1, 1).to(device)
        return torch.cat([reg_emb, token_emb], dim=1)


class MLPMixerLayer(nn.Module):
    def __init__(self, n_tokens, embed_dim, token_dim, channel_dim, dropout_rate, drop_path_rate=0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(n_tokens, token_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(token_dim, n_tokens),
            nn.Dropout(dropout_rate)
        )
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(embed_dim, channel_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(channel_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, mask=None):
        residual = x
        x = self.token_norm(x)
        x = x.transpose(1, 2)
        if mask is not None:
            x = x * mask.unsqueeze(1).float()
        x = self.token_mix(x)
        x = x.transpose(1, 2)
        x = residual + self.drop_path1(x)

        residual = x
        x = self.channel_norm(x)
        x = self.channel_mix(x)
        x = residual + self.drop_path2(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, n_tokens, embed_dim, token_dim, channel_dim, num_layers, dropout_rate, drop_path_rate=0.0):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList([
            MLPMixerLayer(n_tokens, embed_dim, token_dim, channel_dim, dropout_rate, dpr[i])
            for i in range(num_layers)
        ])

    def forward(self, x):
        batch_size, n_actual, _ = x.shape
        mask = None
        if n_actual < self.n_tokens:
            pad = torch.zeros(batch_size, self.n_tokens - n_actual, self.embed_dim,
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
            mask = torch.ones(batch_size, self.n_tokens, device=x.device, dtype=torch.bool)
            mask[:, n_actual:] = False
        for layer in self.layers:
            x = layer(x, mask)
        return x[:, 0, :]


class DeepLearningRegressor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.nn_model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, reg_embedding):
        return self.nn_model(reg_embedding).squeeze(-1)


class TransTabRegressor(nn.Module):
    def __init__(self, nlp_processor, embed_dim=64, hidden_dim=256,
                 num_layers=2, dropout_rate=0.05, n_tokens=12, value_hidden_dim=128, drop_path_rate=0.1):
        super().__init__()
        self.config = {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'n_tokens': n_tokens,
            'value_hidden_dim': value_hidden_dim,
            'drop_path_rate': drop_path_rate
        }
        self.nlp_processor = nlp_processor
        self.mlp_mixer = MLPMixer(
            n_tokens=n_tokens,
            embed_dim=embed_dim,
            token_dim=hidden_dim // 2,
            channel_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate
        )
        self.deep_learning_model = DeepLearningRegressor(
            embed_dim, hidden_dim, dropout_rate
        )
        self.best_r2 = -100.0

    def forward(self, feature_names, feature_values):
        emb = self.nlp_processor(feature_names, feature_values)
        reg_emb = self.mlp_mixer(emb)
        pred = self.deep_learning_model(reg_emb)
        return pred


# ===================== Model Loading & Prediction Functions =====================
@st.cache_resource
def load_trained_model(model_path="bho_optimized_model.pth"):
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    nlp_processor = NLPDataProcessor(
        embed_dim=cfg['embed_dim'],
        value_hidden_dim=cfg['value_hidden_dim']
    )
    nlp_processor.feature_name_vectorizer.vocabulary_ = checkpoint['vocab']
    nlp_processor.name_projection = nn.Linear(len(checkpoint['vocab']), cfg['embed_dim'])

    model = TransTabRegressor(nlp_processor=nlp_processor, **cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


# Categorical feature encoding mapping (consistent with training labels)
CONSTRAINT_MAP = {"Fixed-Fixed": 0, "Fixed-Simply Supported": 1, "Simply Supported-Simply Supported": 2}
SHAPE_MAP = {"Circular": 0, "Square": 1}


def predict_displacement(model, checkpoint, input_params):
    device = next(model.parameters()).device
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    feature_names = checkpoint['feature_names']

    input_data = input_params.copy()
    input_data['end_constraint'] = CONSTRAINT_MAP[input_data['end_constraint']]
    input_data['cross_sectional_shape'] = SHAPE_MAP[input_data['cross_sectional_shape']]

    feature_values = np.array([[input_data[name] for name in feature_names]], dtype=np.float32)
    feature_values_scaled = scaler_x.transform(feature_values)
    feature_tensor = torch.tensor(feature_values_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_scaled = model([feature_names], feature_tensor)
    pred = scaler_y.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()[0]
    return float(pred)


# ===================== Main Interface Logic =====================
# 1. Load model
with st.spinner("Loading model, please wait..."):
    model, checkpoint = load_trained_model()
st.success("✅ Model loaded successfully, ready for prediction")

# 2. Sidebar parameter input form
with st.sidebar:
    st.header("Parameter Input")
    st.subheader("Categorical Features")
    end_constraint = st.selectbox("End Constraint Type", options=list(CONSTRAINT_MAP.keys()), index=0)
    cross_sectional_shape = st.selectbox("Cross-section Shape", options=list(SHAPE_MAP.keys()), index=0)

    st.subheader("Geometric and Material Parameters")
    cross_sectional_dimension = st.number_input("Cross-section Dimension (mm)", min_value=50.0, max_value=500.0,
                                                value=140.0, step=1.0)
    clear_span = st.number_input("Clear Span (mm)", min_value=500.0, max_value=3000.0, value=1400.0, step=10.0)
    steel_tube_wall_thickness = st.number_input("Steel Tube Wall Thickness (mm)", min_value=1.0, max_value=20.0,
                                                value=4.0, step=0.5)
    concrete_compressive_strength = st.number_input("Concrete Compressive Strength (MPa)", min_value=20.0,
                                                    max_value=100.0, value=30.0, step=1.0)
    steel_tube_yield_strength = st.number_input("Steel Tube Yield Strength (MPa)", min_value=200.0, max_value=600.0,
                                                value=235.0, step=5.0)
    recycled_ratio = st.number_input("Recycled Coarse Aggregate Replacement Ratio", min_value=0.0, max_value=1.0,
                                     value=0.5, step=0.05)

    st.subheader("Load Parameters")
    axial_load_ratio = st.number_input("Axial Load Ratio", min_value=0.0, max_value=0.6, value=0.3, step=0.05)
    drop_height = st.number_input("Drop Hammer Height (m)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    impact_mass = st.number_input("Impact Mass (kg)", min_value=10.0, max_value=500.0, value=50.0, step=1.0)

    predict_btn = st.button("Start Prediction", type="primary", use_container_width=True)

# 3. Prediction result display
if predict_btn:
    input_params = {
        "end_constraint": end_constraint,
        "cross_sectional_shape": cross_sectional_shape,
        "cross_sectional_dimension": cross_sectional_dimension,
        "clear_span": clear_span,
        "axial_load_ratio": axial_load_ratio,
        "recycled_coarse_aggregate_replacement_ratio": recycled_ratio,
        "steel_tube_wall_thickness": steel_tube_wall_thickness,
        "concrete_compressive_strength": concrete_compressive_strength,
        "steel_tube_yield_strength": steel_tube_yield_strength,
        "drop_height": drop_height,
        "impact_mass": impact_mass
    }

    with st.spinner("Calculating..."):
        result = predict_displacement(model, checkpoint, input_params)

    st.subheader("Prediction Result")
    st.metric(label="Displacement at Impact Point", value=f"{result:.4f} mm")

    with st.expander("View input parameter details"):
        st.json(input_params)