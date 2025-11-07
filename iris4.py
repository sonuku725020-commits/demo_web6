import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# -----------------------------------------------------------
# 🧠 Title and Description
# -----------------------------------------------------------
st.set_page_config(page_title="Bagging Ensemble Dashboard", layout="wide")
st.title("🎯 Bagging Ensemble Classifier Dashboard")
st.write(
    """This dashboard trains a **Bagging Ensemble** using `scikit-learn`.
    It visualizes the decision surface, confusion matrix, and performance metrics."""
)

# -----------------------------------------------------------
# ⚙️ Sidebar Controls
# -----------------------------------------------------------
st.sidebar.header("Configure Model")

n_estimators = st.sidebar.slider("Number of Trees", 5, 100, 30, 5)
max_samples = st.sidebar.slider("Max Samples (fraction)", 0.1, 1.0, 0.8, 0.1)
noise = st.sidebar.slider("Data Noise", 0.0, 0.5, 0.1, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 999, 42)

# -----------------------------------------------------------
# 📊 Generate Synthetic Data
# -----------------------------------------------------------
X, y = make_classification(
    n_samples=400,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5 - noise,
    random_state=random_state
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

# -----------------------------------------------------------
# 🔍 Train Bagging Classifier
# -----------------------------------------------------------
base_estimator = DecisionTreeClassifier(random_state=random_state)
bag_clf = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=n_estimators,
    max_samples=max_samples,
    bootstrap=True,
    random_state=random_state
)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.metric("Accuracy", f"{acc:.3f}")

# -----------------------------------------------------------
# 🧩 Visualizations
# -----------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ✅ Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    bag_clf,
    X_train,
    response_method="predict",
    cmap="RdBu",
    alpha=0.6,
    ax=axes[0]
)
axes[0].scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    edgecolor="k",
    cmap="RdBu",
)
axes[0].set_title("Decision Boundary")

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted labels")
axes[1].set_ylabel("True labels")

st.pyplot(fig)

# -----------------------------------------------------------
# 📜 Summary Report
# -----------------------------------------------------------
st.subheader("Summary")
st.write(f"""
- **Training Samples:** {len(X_train)}  
- **Test Samples:** {len(X_test)}  
- **Number of Base Estimators:** {n_estimators}  
- **Max Samples per Estimator:** {max_samples}  
- **Random State:** {random_state}  
""")

st.markdown("---")
st.info("🌟 Tip: Adjust sliders in the sidebar to explore how ensemble size and sampling fraction affect accuracy and boundaries!")

# Just enough whimsy:
st.write("🤖 *Ensembles: because one tree is nice, but a whole forest parties harder!*")