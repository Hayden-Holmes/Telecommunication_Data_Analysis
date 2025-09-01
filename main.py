print ("SCript started")


from pyexpat import features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from data_clean import transform_dataset
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree
from fpdf import FPDF
import os
import img2pdf
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


FILE_HEADERS = {
    "Logistic Regression": ["LR_confusion_report", "LR_feature_importance"],
    "Decision Tree": ["DT_confusion_report", "DT_feature_importance"],
    "Random Forest": ["RF_confusion_report", "RF_feature_importance"]
}
DPI_VALUE = 150

all_metrics = []  # global list to hold all metrics


def record_metrics(y_test, y_pred, y_prob, segment, model, n_total):
    report = classification_report(
        y_test, y_pred, target_names=["Stay","Churn"], output_dict=True
    )
    auc = roc_auc_score(y_test, y_prob)

    row = {
        "segment": segment,
        "model": model,
        "n_customers": n_total,                         # full subset size
        "n_test": len(y_test),                          # test set size
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "roc_auc": auc
    }

    all_metrics.append(row)  # append to global list
    return row

#method to record fetures in all_features
all_features = []

def record_feature_importances(importances, features, segment, model):
    for feat, val in zip(features, importances):
        all_features.append({
            "segment": segment,
            "model": model,
            "feature": feat,
            "importance": float(val)  # coef for LR, feature_importances_ for trees
        })

def plot_confusion_and_report(y_test, y_pred, y_prob, segment, model):

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report_dict = classification_report(y_test, y_pred, target_names=["Stay","Churn"], output_dict=True)

    # Extract summary (weighted avg + accuracy)
    summary = {
        "accuracy": report_dict["accuracy"],
        "precision": report_dict["weighted avg"]["precision"],
        "recall": report_dict["weighted avg"]["recall"],
        "f1-score": report_dict["weighted avg"]["f1-score"]
    }
    df = pd.DataFrame([summary])

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # Left: confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Stay","Pred Churn"],
                yticklabels=["Actual Stay","Actual Churn"],
                ax=axes[0])
    axes[0].set_title(f"{model} – Confusion Matrix")

    # Right: summary table
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=df.round(2).values,
        colLabels=df.columns,
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.2)
    axes[1].set_title(f"Summary (AUC={auc:.3f})")

    plt.tight_layout()
    fname = f"graphics/{segment}/{model}_confusion_report.png"
    plt.savefig(fname, dpi=DPI_VALUE, bbox_inches="tight")
    plt.close()
    print(f"{model} combined confusion + report saved as {fname}")


def logistic_regression_modeling(df, segment, drop=None):
    X, y, X_train, X_test, y_train, y_test, clf, ordinal_categoricals,binary_categoricals_2level,continuous_and_counts, X_train_df = transform_dataset(df, clf=True, drop=drop)

    # logistic_regression
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    clf.fit(X_train, y_train)

    #Evaluate the model
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]


    

    #Begin work for coefficients (Feature Importance) graphic
    feature_names = X_train_df.columns
    importances = clf.named_steps["model"].coef_[0]
    indices = np.argsort(np.abs(importances))[::-1]  # sort by absolute impact

    plt.figure(figsize=(10,6))
    sns.barplot(
        x=importances[indices],
        y=feature_names[indices],
        palette="coolwarm",
        hue=importances[indices],
    
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Logistic Regression – Feature Importance ({segment})")
    plt.xlabel("Coefficient (Log-Odds Impact)")
    plt.ylabel("Features")
    plt.tight_layout()
    fname = f"graphics/{segment}/LR_feature_importance.png"
    plt.savefig(fname, dpi=DPI_VALUE)
    plt.close()
    print("LR Feature importance saved as", fname)

    
    
    
    
    
    #classification report + roc score graphic
    plot_confusion_and_report(y_test, y_pred, y_prob, segment, "LR")

    # Record metrics and features
    record_metrics(y_test, y_pred, y_prob, segment, "LR", len(y_train)+len(y_test))
    coefs = clf.named_steps["model"].coef_[0]
    record_feature_importances(coefs, X_train_df.columns, segment, "Logistic Regression")



def decision_tree_modeling(df, segment, drop=None):
    X,y,X_train, X_test, y_train, y_test, clf,ordinal_categoricals,binary_categoricals_2level,continuous_and_counts, X_train_df = transform_dataset(df, clf=True, drop=drop)

    dt = DecisionTreeClassifier(
        criterion="gini",       
        max_depth=4,             # shallower = simpler, more general rules
        min_samples_split=50,    # don’t split unless you have 50 samples
        min_samples_leaf=20,     # each leaf must have at least 20 customers
        class_weight="balanced", # handle class imbalance
        random_state=42
    )

    dt.fit(X_train, y_train)

    # predictions
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:,1]


  
    #visualize tree
    n_features = X.shape[1]
    depth = 3  
    width = min(2 * n_features, 40)   # keep within 40 inches
    height = min(2 * depth, 20)       # keep within 20 inches

    plt.figure(figsize=(width, height))
    plot_tree(dt, filled=True, feature_names=continuous_and_counts + binary_categoricals_2level + ordinal_categoricals,max_depth=depth, class_names=["Stay", "Churn"], rounded=True, fontsize=10)
    plt.title("Decision Tree – "+segment)
    plt.savefig("graphics/"+segment+"/DT_decision_tree.png", dpi=DPI_VALUE)
    plt.close()
    print("DT Decision tree saved as graphics/"+segment+"/DT_decision_tree.png")

    #Feature importance
    feature_importances = dt.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(12,8))
    sns.barplot(x=feature_importances[indices], y=np.array(continuous_and_counts + binary_categoricals_2level + ordinal_categoricals)[indices])
    plt.title("Feature Importance – Decision Tree")
    plt.xlabel("Relative Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("graphics/"+segment+"/DT_feature_importance.png", dpi=DPI_VALUE)
    plt.close()
    print("DT Feature importance saved as graphics/"+segment+"/DT_feature_importance.png")

    #classification report + roc score graphic
    plot_confusion_and_report(y_test, y_pred_dt, y_prob_dt, segment, "DT")

    # Record metrics and features
    record_metrics(y_test, y_pred_dt, y_prob_dt, segment, "DT", len(y_train)+len(y_test))
    record_feature_importances(dt.feature_importances_, X_train_df.columns, segment, "Decision Tree")



def random_forest_modeling(df, segment, drop):
    X, y, X_train, X_test, y_train, y_test, clf, continuous_and_counts, binary_categoricals_2level, ordinal_categoricals, X_train_df = transform_dataset(df, clf=True, drop=drop)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )

    rf.fit(X_train, y_train)
    # Always get the exact feature order the model used
    feature_names = getattr(rf, "feature_names_in_", None)
    if feature_names is None:
        # fallback if sklearn is old or X_train wasn't a DataFrame
        feature_names = np.array(X_train.columns)

    feature_importances = rf.feature_importances_

    # Sanity checks
    assert len(feature_importances) == len(feature_names), \
        f"length mismatch: {len(feature_importances)} vs {len(feature_names)}"

    # Sort once and reuse everywhere
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_names = feature_names[sorted_idx]
    sorted_imps  = feature_importances[sorted_idx]

    # predictions
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:,1]

    #Visualize last tree
    # pick the last tree in the forest
    estimator = rf.estimators_[-1]

    n_features = X.shape[1]
    depth = 3
    width = min(2 * n_features, 40)
    height = min(2 * depth, 20)

    plt.figure(figsize=(width, height))
    plot_tree(
        estimator,
        feature_names=X.columns,
        class_names=["Stay", "Churn"],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=depth
    )
    plt.savefig("graphics/"+segment+"/RF_Main.png")
    plt.close()
    print("Tree saved as graphics/"+segment+"/RF_Main.png")

    

  
   

    # Plot feature importance
    plt.figure(figsize=(12,8))
    sns.barplot(x=sorted_imps, y=sorted_names)
    plt.title(f"Random Forest – Feature Importance – {segment}")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(f"graphics/{segment}/RF_feature_importance.png", dpi=DPI_VALUE)
    plt.close()

    #plot confusion matrix and classification report
    plot_confusion_and_report(y_test, y_pred_rf, y_prob_rf, segment, "RF")

    # Save CSV consistently
    record_feature_importances(sorted_imps, sorted_names, segment, "Random Forest")
    record_metrics(y_test, y_pred_rf, y_prob_rf, segment, "RF", len(y_train)+len(y_test))


def run_models(df, segment, drop=None):
    logistic_regression_modeling(df, segment, drop=drop)
    decision_tree_modeling(df, segment, drop=drop)
    random_forest_modeling(df, segment, drop=drop)


def export_to_pdf(segment):
    print("Exporting graphics to PDF for segment:", segment)

    # Images are inside graphics/{segment}
    img_folder = Path(f"graphics/{segment}")
    imgs = sorted(img_folder.glob("*.png"))

    # Save the PDF one level up: graphics/{segment}_report.pdf
    pdf_path = img_folder.parent / f"Reports/{segment}_report.pdf"

    if not imgs:
        print("No images found for", segment)
        return

    try:
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert([str(img) for img in imgs]))
        print("PDF created successfully.")
    except Exception as e:
        print("Error creating PDF:", e)

    print("PDF saved:", pdf_path)

def comparison_dashboard():
    print("Creating comparison dashboard...")

    # Load CSVs
    metrics_df = pd.read_csv("graphics/Reports/metrics_summary.csv").drop_duplicates()
    features_df = pd.read_csv("graphics/Reports/feature_importances.csv")
    features_df["abs_importance"] = features_df["importance"].abs()

    # Save location
    reports_folder = Path("graphics/Reports")
    reports_folder.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_folder / "comparison_report.pdf"
    print("Saving dashboard to:", pdf_path.resolve())

    with PdfPages(pdf_path) as pdf:
        print("Generating dashboard pages...")
        # Page 1: ROC-AUC heatmap
        
        # Pivot data    
        pivot_auc = metrics_df.pivot(index="segment", columns="model", values="roc_auc")

        # Add n= to segment labels
        segment_counts = metrics_df.groupby("segment")["n_customers"].max().to_dict()
        pivot_auc.index = [f"{seg} (n={segment_counts.get(seg, '?')})" for seg in pivot_auc.index]

        # Create heatmap
        plt.figure(figsize=(6,3))  # smaller figure for tighter boxes
        ax = sns.heatmap(
            pivot_auc,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=False,              # 2) remove legend/colorbar
            linewidths=0.5,          # small gridlines to separate boxes
            linecolor="white",
            vmin=0.75, vmax=1.0      # 4) color scale shifted toward 1
        )

        plt.title("ROC-AUC by Segment and Model", fontsize=12, fontweight="bold")

        # Format labels
        # Move x-axis labels to the top
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        plt.xticks(rotation=0)

        # Remove y-axis label ("Segment") → let row labels act like table rows
        ax.set_ylabel("")

        # Center title above the whole figure
        plt.title("ROC-AUC by Segment and Model", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig("graphics/comparisons/roc_auc_heatmap.png", dpi=DPI_VALUE)
        pdf.savefig()
        plt.close()
        
        # Page 2: Overall model comparison
        overall_perf = (
            metrics_df.groupby("model")[["accuracy","precision","recall","f1","roc_auc"]]
            .mean().reset_index()
        )
        plt.figure(figsize=(6,4))
        sns.barplot(data=overall_perf, x="model", y="roc_auc", palette="viridis", hue="model")
        plt.title("Average ROC-AUC Across Segments")
        plt.ylabel("Mean ROC-AUC")
        plt.ylim(.8, 1)
        plt.tight_layout()
        pdf.savefig()
        plt.savefig("graphics/comparisons/average_roc_auc.png", dpi=DPI_VALUE)
        plt.close()

        # Pages 3+: Top RF features by segment
        for seg in features_df["segment"].unique():
            feat_example = features_df[(features_df["segment"]==seg) & (features_df["model"]=="Random Forest")]
            if feat_example.empty:
                continue
            feat_example = feat_example.sort_values("abs_importance", ascending=False).head(10)

            plt.figure(figsize=(8,5))
            sns.barplot(data=feat_example, x="abs_importance", y="feature", palette="mako", hue="feature")
            plt.title(f"Top Features – {seg} (Random Forest)")
            plt.xlabel("Importance (abs value)")
            plt.tight_layout()
            pdf.savefig()
            plt.savefig(f"graphics/comparisons/top_features_{seg.replace(' ','_')}.png", dpi=DPI_VALUE)
            plt.close()

    print("Dashboard saved as", pdf_path.resolve())


def main(remake=False):

    print("Loading dataset...")
    data_set = pd.read_csv("data/customer_churn.csv")
    segments = {"Complaint_Customers" : (data_set["Complaints"]==1, "Complaints"),
                "No_Complaint_Customers" : (data_set["Complaints"]==0, "Complaints"),
                "Active_Customers" : (data_set["Status"]==1, "Status"),
                "Inactive_Customers" : (data_set["Status"]==2, "Status"),
                "All_Customers" : (None, None)
               }
    #make directory for graphics
    if remake:
        for segment in segments.keys():
            folder = f"graphics/{segment}"
            os.makedirs(folder, exist_ok=True)
            try:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            except Exception as e:
                print("Error clearing directory:", e)
        for name, (condition, col_to_drop) in segments.items():
            if condition is None:
                print(f"Processing segment: {name} (all customers)")
                subset = data_set.copy()
                print(f"Running models for: {name} (n={len(subset)})")
                run_models(subset, name)

            else:
                print(f"Processing segment: {name}")
                subset = data_set.loc[condition].copy()
                print("  attempting to drop column:", col_to_drop)
                print(f"Running models for: {name} (n={len(subset)})")
                run_models(subset, name, col_to_drop)
    else:
        print("Leaving existing graphics intact.")

    # Run models for each segment
    

    # Export charts to PDF
    for seg in segments.keys():
         export_to_pdf(seg)

    # export results to a CSV file
    if remake:
        print("Exporting all results to CSV")
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv("graphics/Reports/metrics_summary.csv", index=False)

        # Save feature importances
        print("Exporting feature importances to CSV")
        df_features = pd.DataFrame(all_features)
        df_features.to_csv("graphics/Reports/feature_importances.csv", index=False)

    #create comparison dashboard
    print("Generating comparison dashboard...")
    comparison_dashboard()

print("Saved metrics and feature importances to graphics/Reports/")

if __name__ == "__main__":
    main(remake=True)
    # main(remake=False)

