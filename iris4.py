import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Gradient Boosting Regression", layout="wide")
st.title("🚀 Gradient Boosting Regression (scikit-learn)")

sns.set_style("whitegrid")

# ------------------------- Sidebar: Data -------------------------
with st.sidebar:
    st.header("1) Data")
    dataset = st.selectbox(
        "Choose dataset",
        ["California Housing", "Diabetes", "Synthetic (make_regression)", "Upload CSV"],
        index=0
    )
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 42, step=1)

@st.cache_data(show_spinner=False)
def load_builtins(name: str):
    if name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        X = data.frame.drop(columns=[data.target_names[0]])
        y = data.frame[data.target_names[0]]
        return X, y, "California Housing"
    elif name == "Diabetes":
        data = load_diabetes(as_frame=True)
        X = data.frame.drop(columns=["target"])
        y = data.frame["target"]
        return X, y, "Diabetes"
    else:
        raise ValueError("Unknown dataset.")

# Dataset handling
if dataset in ["California Housing", "Diabetes"]:
    X, y, ds_name = load_builtins(dataset)
elif dataset == "Synthetic (make_regression)":
    with st.sidebar:
        st.subheader("Synthetic settings")
        n_samples = st.slider("n_samples", 200, 10000, 1500, 100)
        n_features = st.slider("n_features", 2, 100, 12, 1)
        n_informative = st.slider("n_informative", 1, n_features, min(10, n_features), 1)
        noise = st.slider("noise", 0.0, 50.0, 5.0, 0.5)
        bias = st.slider("bias", -10.0, 10.0, 0.0, 0.5)
    X_arr, y_arr = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        bias=bias,
        random_state=random_state,
    )
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(n_features)])
    y = pd.Series(y_arr, name="target")
    ds_name = "Synthetic (make_regression)"
else:
    st.sidebar.subheader("Upload CSV")
    file = st.sidebar.file_uploader("Choose CSV", type=["csv"])
    if file is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = pd.read_csv(file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. Please upload numeric data.")
        st.stop()
    target_col = st.sidebar.selectbox("Target column", numeric_cols, index=len(numeric_cols)-1)
    feature_cols = [c for c in numeric_cols if c != target_col]
    if not feature_cols:
        st.error("No feature columns available after selecting target.")
        st.stop()
    X = df[feature_cols]
    y = df[target_col]
    ds_name = "Uploaded CSV"

# Basic cleaning
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]
if y.isna().any():
    good_idx = ~y.isna()
    X = X.loc[good_idx]
    y = y.loc[good_idx]

# ------------------------- Sidebar: Model -------------------------
with st.sidebar:
    st.header("2) Model (GradientBoostingRegressor)")
    loss = st.selectbox("loss", ["squared_error", "absolute_error", "huber", "quantile"], index=0)
    learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)
    n_estimators = st.slider("n_estimators", 50, 1000, 300, 10)
    subsample = st.slider("subsample", 0.1, 1.0, 1.0, 0.05)
    max_depth = st.slider("max_depth", 1, 15, 3, 1)
    min_samples_split = st.slider("min_samples_split", 2, 100, 2, 1)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 100, 1, 1)

    max_features_mode = st.selectbox("max_features mode", ["None", "sqrt", "log2", "fraction"], index=0)
    if max_features_mode == "fraction":
        max_features = st.slider("max_features (fraction)", 0.1, 1.0, 1.0, 0.05)
    elif max_features_mode in ["sqrt", "log2"]:
        max_features = max_features_mode
    else:
        max_features = None

    criterion = st.selectbox("criterion", ["friedman_mse", "squared_error"], index=0)
    alpha = st.slider("alpha (for huber/quantile)", 0.01, 0.99, 0.9, 0.01)

    early = st.checkbox("Enable early stopping", value=False)
    if early:
        validation_fraction = st.slider("validation_fraction", 0.05, 0.5, 0.1, 0.01)
        n_iter_no_change = st.slider("n_iter_no_change", 2, 50, 10, 1)
        tol = st.select_slider("tol", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    else:
        validation_fraction = 0.1
        n_iter_no_change = None
        tol = 1e-4

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Build model
params = dict(
    loss=loss,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    subsample=subsample,
    criterion=criterion,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    max_features=max_features,
    random_state=random_state,
    validation_fraction=validation_fraction,
    n_iter_no_change=n_iter_no_change,
    tol=tol,
    alpha=alpha,
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Dataset")
    st.write(f"- Name: {ds_name}")
    st.write(f"- Samples: {len(X)} | Features: {X.shape[1]}")
    st.dataframe(X.head())
with right:
    st.subheader("Train")
    go = st.button("Train model", type="primary", use_container_width=True)

if not go:
    st.info("Adjust options in the sidebar and click 'Train model'.")
    st.stop()

with st.spinner("Training..."):
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # quick 5-fold CV on all data
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)

# ------------------------- Metrics -------------------------
st.subheader("Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² (Train)", f"{r2_train:.3f}")
m2.metric("R² (Test)", f"{r2_test:.3f}")
m3.metric("RMSE (Test)", f"{rmse_test:.3f}")
m4.metric("MAE (Test)", f"{mae_test:.3f}")
st.caption(f"5-fold CV R²: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

# ------------------------- Visualizations -------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Predicted vs Actual", "Residuals", "Feature Importance", "Training Curve"]
)

with tab1:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred_test, alpha=0.6, edgecolor="k")
    vmin = float(min(y_test.min(), y_pred_test.min()))
    vmax = float(max(y_test.max(), y_pred_test.max()))
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=2, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (Test)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab2:
    residuals = y_test - y_pred_test
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(y_pred_test, residuals, alpha=0.6, edgecolor="k")
        ax1.axhline(0, color="r", linestyle="--")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Residual")
        ax1.set_title("Residuals vs Predicted")
        st.pyplot(fig1, clear_figure=True)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title("Residual Distribution")
        st.pyplot(fig2, clear_figure=True)

with tab3:
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write(importances.to_frame("importance"))
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.head(min(20, len(importances))).iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_title("Top Feature Importances")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Model does not expose feature_importances_.")

with tab4:
    # Staged MSE over boosting iterations
    train_mse, test_mse = [], []
    for yp in model.staged_predict(X_train):
        train_mse.append(mean_squared_error(y_train, yp))
    for yp in model.staged_predict(X_test):
        test_mse.append(mean_squared_error(y_test, yp))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_mse, label="Train MSE", alpha=0.8)
    ax.plot(test_mse, label="Test MSE", alpha=0.8)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("MSE")
    ax.set_title("Training Curve (MSE by stage)")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

st.markdown("---")
st.subheader("Summary")
st.write(f"- Dataset: {ds_name}")
st.write(f"- Train samples: {len(X_train)} | Test samples: {len(X_test)}")
st.write(f"- Features: {list(X.columns)}")
st.caption("Note: We report R² as 'accuracy' for regression.")
