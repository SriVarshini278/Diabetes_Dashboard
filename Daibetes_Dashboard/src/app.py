import os
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
from data_utils import load_data
from features import add_bmi_category
from model import train_model
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Insights", "Model Training", "Reports"])

def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_figure(fig, fname):
    path = os.path.join(FIGURES_DIR, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def _save_model(obj, fname):
    path = os.path.join(REPORTS_DIR, fname)
    joblib.dump(obj, path)
    return path

def generate_text_report(y_true, y_pred, accuracy, path_txt):
    report = classification_report(y_true, y_pred, digits=4)
    conf = confusion_matrix(y_true, y_pred)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(f"Model Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf) + "\n")
    return path_txt

def generate_csv_predictions(df, y_pred, path_csv):
    df_out = df.copy()
    df_out["Prediction"] = y_pred
    df_out.to_csv(path_csv, index=False)
    return path_csv

def generate_excel_predictions(df, y_pred, path_xlsx):
    df_out = df.copy()
    df_out["Prediction"] = y_pred
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Predictions")
    return path_xlsx

def generate_pdf(title, accuracy, images, path_pdf):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(title, styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Model Accuracy: {accuracy:.4f}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    for caption, img_path in images:
        elements.append(Paragraph(caption, styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Image(img_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    doc.build(elements)
    with open(path_pdf, "wb") as f:
        f.write(buffer.getbuffer())
    buffer.close()
    return path_pdf

@st.cache_data
def load_default_data():
    return load_data()

df = load_default_data()

if page == "Home":
    st.title("📊 Diabetes Prediction Dashboard")
    st.markdown(
        """
        Welcome to the **Diabetes Prediction Dashboard**.  
        
        ### How to Use:
        1. **Upload Data** → Use your own CSV or default dataset  
        2. **Insights** → Explore patterns, distributions, and correlations  
        3. **Model Training** → Train a logistic regression model and evaluate performance  
        4. **Reports** → Access auto-generated reports (PDF, CSV, Excel, TXT, model, and figures)  

        All reports are automatically saved in the **reports/** folder.  
        """
    )

if page == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Custom dataset loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
    st.subheader("Preview")
    st.dataframe(df.head())

if page == "Insights":
    st.header("Exploratory Data Analysis")
    st.subheader("Dataset Overview")
    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.write(df.describe(include="all"))
    st.subheader("Outcome Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df, ax=ax1)
    ax1.set_title("Distribution of Diabetes Outcomes")
    st.pyplot(fig1)
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax2)
    ax2.set_title("Feature Correlation")
    st.pyplot(fig2)
    st.subheader("Glucose vs BMI by Outcome")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="Glucose", y="BMI", hue="Outcome", data=df, ax=ax3)
    ax3.set_title("Glucose vs BMI")
    st.pyplot(fig3)
    st.subheader("Age vs Outcome Boxplot")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Outcome", y="Age", data=df, ax=ax4)
    ax4.set_title("Age vs Outcome")
    st.pyplot(fig4)
    if st.button("Save Insight Figures"):
        t = _timestamp()
        _save_figure(fig1, f"class_distribution_{t}.png")
        _save_figure(fig2, f"correlation_{t}.png")
        _save_figure(fig3, f"glucose_bmi_{t}.png")
        _save_figure(fig4, f"age_boxplot_{t}.png")
        st.success("EDA figures saved in reports/figures")

if page == "Model Training":
    st.header("Train & Evaluate Model")
    df_train = add_bmi_category(df.copy())
    if "Outcome" not in df_train.columns:
        st.error("Dataset must contain an Outcome column")
    else:
        try:
            if "BMI_Category" in df_train.columns:
                df_train = pd.get_dummies(df_train, columns=["BMI_Category"], drop_first=True)
            categorical_cols = df_train.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
            X = df_train.drop("Outcome", axis=1)
            y = df_train["Outcome"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            model, train_acc = train_model(pd.concat([X_train, y_train], axis=1))
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            st.write(f"Train Accuracy: {train_acc:.4f}")
            st.success(f"Test Accuracy: {test_acc:.4f}")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            st.pyplot(fig_cm)
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)
            if st.button("Generate Reports"):
                t = _timestamp()
                model_path = _save_model(model, f"model_{t}.joblib")
                txt_path = os.path.join(REPORTS_DIR, f"classification_report_{t}.txt")
                csv_path = os.path.join(REPORTS_DIR, f"predictions_{t}.csv")
                xlsx_path = os.path.join(REPORTS_DIR, f"predictions_{t}.xlsx")
                cm_path = _save_figure(fig_cm, f"confusion_matrix_{t}.png")
                roc_path = _save_figure(fig_roc, f"roc_curve_{t}.png")
                generate_text_report(y_test, y_pred, test_acc, txt_path)
                generate_csv_predictions(pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1), y_pred, csv_path)
                generate_excel_predictions(pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1), y_pred, xlsx_path)
                pdf_path = os.path.join(REPORTS_DIR, f"report_{t}.pdf")
                images = [("Confusion Matrix", cm_path), ("ROC Curve", roc_path)]
                generate_pdf("Diabetes Prediction Report", test_acc, images, pdf_path)
                st.success("Reports generated in reports/ folder")
                st.write(f"Model: {model_path}")
                st.write(f"PDF: {pdf_path}")
                st.write(f"CSV: {csv_path}")
                st.write(f"Excel: {xlsx_path}")
                st.write(f"Text: {txt_path}")
        except Exception as e:
            st.error(f"Error: {e}")

if page == "Reports":
    st.header("Saved Reports")
    for root, _, files in os.walk(REPORTS_DIR):
        for f in sorted(files, reverse=True):
            fp = os.path.join(root, f)
            if not os.path.isfile(fp):
                continue
            st.write(f)
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                st.image(fp, use_column_width=True)
            with open(fp, "rb") as fh:
                st.download_button(f"Download {f}", fh, file_name=f)
