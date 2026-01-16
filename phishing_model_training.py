import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress joblib memory warnings (only affects RandomizedSearchCV)
import logging
logging.getLogger('joblib').setLevel(logging.ERROR)

# Setting style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using GradientBoosting instead.")

# ==================== LOAD DATASETS ====================
def load_datasets(train_path='train.csv', val_path='val.csv', test_path='test.csv'):
    """Load preprocessed datasets"""
    import os
    
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    
    # Check if files exist
    for path, name in [(train_path, 'Training'), (val_path, 'Validation'), (test_path, 'Test')]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {name} file not found: {path}")
            print(f"[INFO] Current directory: {os.getcwd()}")
            print(f"[INFO] Files in current directory:")
            for file in os.listdir('.'):
                if file.endswith('.csv'):
                    print(f"    - {file}")
            raise FileNotFoundError(f"Required file '{path}' not found. Please ensure all CSV files are in the correct location.")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"[+] Train set: {train_df.shape}")
    print(f"[+] Validation set: {val_df.shape}")
    print(f"[+] Test set: {test_df.shape}")
    
    # Separate features and labels
    X_train = train_df.drop(['URL', 'Label'], axis=1)
    y_train = train_df['Label']
    
    X_val = val_df.drop(['URL', 'Label'], axis=1)
    y_val = val_df['Label']
    
    X_test = test_df.drop(['URL', 'Label'], axis=1)
    y_test = test_df['Label']
    
    print(f"\n[*] Feature columns ({len(X_train.columns)}):")
    print(f"    {list(X_train.columns)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ==================== FEATURE SCALING ====================
def scale_features(X_train, X_val, X_test):
    """Standardize features for models that need scaling"""
    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("[+] Features scaled using StandardScaler")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_confusion_matrix(cm, model_name, dataset_name, save_path):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}\n{dataset_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved: {save_path}")

def plot_metrics_bar(results_dict, model_name, dataset_name, save_path):
    """Plot metrics as bar chart"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    values = [results_dict[m] for m in metrics]
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#06A77D'])
    plt.ylim(0, 1.0)
    plt.title(f'Performance Metrics - {model_name}\n{dataset_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved: {save_path}")

def plot_roc_curve(model, X, y, model_name, dataset_name, save_path, is_scaled=False):
    """Plot ROC curve"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.decision_function(X)
    
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}\n{dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    [+] Saved: {save_path}")

def plot_feature_importance(model, feature_names, model_name, save_path, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='#06A77D')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    [+] Saved: {save_path}")

# ==================== MODEL EVALUATION ====================
def evaluate_model(model, X, y, dataset_name="Dataset", model_name="Model", 
                   feature_names=None, create_plots=True):
    """Comprehensive model evaluation with visualizations"""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{'=' * 70}")
    print(f"{dataset_name.upper()} EVALUATION")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"    TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
    print(f"    FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")
    
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Create visualizations if requested
    if create_plots:
        print(f"\n[*] Generating visualizations for {model_name}...")
        safe_model_name = model_name.replace(" ", "_").lower()
        safe_dataset_name = dataset_name.replace(" ", "_").lower()
        
        # Confusion Matrix
        cm_path = f'plots/{safe_model_name}_{safe_dataset_name}_confusion_matrix.png'
        plot_confusion_matrix(cm, model_name, dataset_name, cm_path)
        
        # Metrics Bar Chart
        metrics_path = f'plots/{safe_model_name}_{safe_dataset_name}_metrics.png'
        results_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        plot_metrics_bar(results_dict, model_name, dataset_name, metrics_path)
        
        # ROC Curve
        roc_path = f'plots/{safe_model_name}_{safe_dataset_name}_roc_curve.png'
        plot_roc_curve(model, X, y, model_name, dataset_name, roc_path)
        
        # Feature Importance (for tree-based models)
        if feature_names is not None and hasattr(model, 'feature_importances_'):
            fi_path = f'plots/{safe_model_name}_feature_importance.png'
            plot_feature_importance(model, feature_names, model_name, fi_path)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

# ==================== BASELINE MODELS ====================
def train_baseline_models(X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled):
    """Train baseline models without hyperparameter tuning"""
    print("\n" + "=" * 70)
    print("TASK 2: TRAINING BASELINE MODELS")
    print("=" * 70)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    models = {}
    results = {}
    feature_names = list(X_train.columns)
    
    # 1. Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    print("[+] Logistic Regression trained")
    
    lr_results = evaluate_model(lr_model, X_val_scaled, y_val, 
                                "Validation Set", "Logistic Regression",
                                feature_names=feature_names)
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = lr_results
    
    # 2. Random Forest
    print("\n[2/3] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("[+] Random Forest trained")
    
    rf_results = evaluate_model(rf_model, X_val, y_val,
                                "Validation Set", "Random Forest",
                                feature_names=feature_names)
    models['Random Forest'] = rf_model
    results['Random Forest'] = rf_results
    
    # 3. XGBoost or Gradient Boosting
    if HAS_XGBOOST:
        print("\n[3/3] Training XGBoost...")
        xgb_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        print("[+] XGBoost trained")
        
        xgb_results = evaluate_model(xgb_model, X_val, y_val,
                                    "Validation Set", "XGBoost",
                                    feature_names=feature_names)
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_results
    else:
        print("\n[3/3] Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        print("[+] Gradient Boosting trained")
        
        gb_results = evaluate_model(gb_model, X_val, y_val,
                                   "Validation Set", "Gradient Boosting",
                                   feature_names=feature_names)
        models['Gradient Boosting'] = gb_model
        results['Gradient Boosting'] = gb_results
    
    return models, results, feature_names

# ==================== MODEL COMPARISON ====================
def compare_models(results):
    """Compare all models and select the best"""
    print("\n" + "=" * 70)
    print("TASK 3: MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1_score'] for r in results.values()]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Plot comparison
    print("\n[*] Generating model comparison visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, comparison_df[metric], width, label=metric, color=color)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/baseline_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    [+] Saved: plots/baseline_model_comparison.png")
    
    # Select best model
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    best_f1 = comparison_df['F1-Score'].max()
    
    print(f"\n[+] BEST MODEL: {best_model_name}")
    print(f"[+] Best F1-Score: {best_f1:.4f}")
    
    return best_model_name, comparison_df

# ==================== HYPERPARAMETER TUNING ====================
def tune_hyperparameters(model_name, X_train, y_train, X_val, y_val, 
                         X_train_scaled=None, X_val_scaled=None, feature_names=None):
    """Task 4: Hyperparameter tuning for the best model"""
    print("\n" + "=" * 70)
    print(f"TASK 4: HYPERPARAMETER TUNING - {model_name}")
    print("=" * 70)
    
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
        X_train_use = X_train_scaled
        X_val_use = X_val_scaled
        
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        X_train_use = X_train
        X_val_use = X_val
        
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        X_train_use = X_train
        X_val_use = X_val
        
    else:  # Gradient Boosting
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
        X_train_use = X_train
        X_val_use = X_val
    
    print(f"[*] Using RandomizedSearchCV for faster tuning...")
    print(f"[*] Parameter grid: {param_grid}")
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='f1',
        n_jobs=2,
        random_state=42,
        verbose=1
    )
    
    print("\n[*] Tuning... (this may take a few minutes)")
    random_search.fit(X_train_use, y_train)
    
    print(f"\n[+] Best parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"    {param}: {value}")
    
    print(f"\n[+] Best cross-validation F1-score: {random_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = random_search.best_estimator_
    tuned_results = evaluate_model(best_model, X_val_use, y_val, 
                                   "Validation Set", f"{model_name} (Tuned)",
                                   feature_names=feature_names)
    
    return best_model, random_search.best_params_, tuned_results

# ==================== FINAL EVALUATION ====================
def final_evaluation(model, X_test, y_test, X_test_scaled=None, 
                    model_name="Model", feature_names=None):
    """Final evaluation on test set"""
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)
    
    if model_name == 'Logistic Regression' and X_test_scaled is not None:
        X_test_use = X_test_scaled
    else:
        X_test_use = X_test
    
    test_results = evaluate_model(model, X_test_use, y_test, 
                                  "Test Set", f"{model_name} (Final)",
                                  feature_names=feature_names)
    
    return test_results

# ==================== SAVE MODEL ====================
def save_model(model, scaler, model_name, best_params, results, comparison_df):
    """Task 5: Save final model and documentation"""
    print("\n" + "=" * 70)
    print("TASK 5: MODEL SAVING & DOCUMENTATION")
    print("=" * 70)
    
    # Save model
    model_filename = f'phishing_detection_model_{model_name.replace(" ", "_").lower()}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Model saved: {model_filename}")
    
    # Save scaler if used
    if scaler is not None:
        scaler_filename = 'feature_scaler.pkl'
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[+] Scaler saved: {scaler_filename}")
    
    # Save documentation
    doc_filename = 'model_documentation.txt'
    with open(doc_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PHISHING DETECTION MODEL - FINAL DOCUMENTATION\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Final Model: {model_name}\n\n")
        
        f.write("Model Comparison:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("Final Test Results:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
        f.write("\n")
        
        f.write("Generated Visualizations:\n")
        f.write("  - Baseline model comparison chart\n")
        f.write("  - Confusion matrices for all models\n")
        f.write("  - Performance metrics bar charts\n")
        f.write("  - ROC curves for all models\n")
        f.write("  - Feature importance plots (for tree-based models)\n")
        f.write("\nAll plots saved in 'plots/' directory\n")
        
    print(f"[+] Documentation saved: {doc_filename}")
    print("\n[SUCCESS] All files saved successfully!")

# ==================== MAIN PIPELINE ====================
def main():
    """Complete ML pipeline for Milestone 2 with visualizations"""
    import os
    
    print("\n" + "=" * 70)
    print("PHISHING DETECTION - MILESTONE 2 PIPELINE")
    print("Machine Learning Model Training & Evaluation with Visualizations")
    print("=" * 70)
    
    # Prompt user for file paths if default files don't exist
    train_path = 'train.csv'
    val_path = 'val.csv'
    test_path = 'test.csv'
    
    if not os.path.exists(train_path):
        print("\n[!] Default CSV files not found in current directory.")
        print(f"[*] Current directory: {os.getcwd()}")
        
        # List CSV files in current directory
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            print(f"[*] Available CSV files:")
            for i, file in enumerate(csv_files, 1):
                print(f"    {i}. {file}")
        
        print("\n[?] Please specify the file paths:")
        train_path = input("Enter path to training CSV file (or press Enter for 'train.csv'): ").strip() or 'train.csv'
        val_path = input("Enter path to validation CSV file (or press Enter for 'val.csv'): ").strip() or 'val.csv'
        test_path = input("Enter path to test CSV file (or press Enter for 'test.csv'): ").strip() or 'test.csv'
    
    # Load datasets
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(train_path, val_path, test_path)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )
    
    # Task 2: Train baseline models
    models, baseline_results, feature_names = train_baseline_models(
        X_train, y_train, X_val, y_val,
        X_train_scaled, X_val_scaled
    )
    
    # Task 3: Compare models
    best_model_name, comparison_df = compare_models(baseline_results)
    
    # Task 4: Hyperparameter tuning
    best_model, best_params, tuned_results = tune_hyperparameters(
        best_model_name, X_train, y_train, X_val, y_val,
        X_train_scaled, X_val_scaled, feature_names
    )
    
    # Final test evaluation
    test_results = final_evaluation(
        best_model, X_test, y_test, X_test_scaled, 
        best_model_name, feature_names
    )
    
    # Task 5: Save model and documentation
    save_model(
        best_model, 
        scaler if best_model_name == 'Logistic Regression' else None,
        best_model_name,
        best_params,
        test_results,
        comparison_df
    )
    
    print("\n" + "=" * 70)
    print("[SUCCESS] MILESTONE 2 COMPLETED!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  1. model_documentation.txt - Complete documentation")
    print("  2. phishing_detection_model_*.pkl - Trained model")
    print("  3. plots/ - Directory with all visualizations")
    print("\nNext Steps:")
    print("  1. Review all visualizations in plots/ directory")
    print("  2. Analyze model performance across metrics")
    print("  3. Check feature importance for insights")
    print("  4. Test model on new URLs")

if __name__ == "__main__":
    main()