import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import joblib

# Set style
plt.style.use('default')  # Use default style
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("viridis")

def plot_feature_importance():
    """Plot feature importance from the trained Random Forest model"""
    print("\nGenerating feature importance visualization...")
    
    try:
        # Load the trained model
        model = joblib.load('models/final_model.joblib')
        
        # Load feature names
        with open('models/final_model_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Most Important Features', fontsize=14)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved feature importance visualization to 'reports/figures/feature_importance.png'")
        return feature_importance_df
        
    except Exception as e:
        print(f"Error generating feature importance plot: {str(e)}")
        return None

def create_visualizations():
    """Generate visualizations for model performance."""
    print("Generating visualizations...")
    
    # Create figures directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load model results
    results_df = pd.read_csv('reports/model_training_results.csv')
    
    # 1. Model Comparison Bar Plot
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    plot_df = results_df.melt(id_vars=['model'], 
                            value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                            var_name='metric', value_name='score')
    
    # Map metric names to display names
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'roc_auc': 'ROC-AUC'
    }
    plot_df['metric'] = plot_df['metric'].map(metric_names)
    
    # Create the plot
    g = sns.catplot(
        data=plot_df, kind='bar',
        x='model', y='score', hue='metric',
        height=6, aspect=1.5, legend_out=False
    )
    
    plt.title('Model Performance Comparison', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('reports/figures/model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Generate model comparison visualization from CSV results
    try:
        # Load model results
        results_df = pd.read_csv('reports/model_training_results.csv')
        
        # 2.1 Model Metrics Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (metric, name) in enumerate(zip(metrics, metrics_names)):
            ax = axes[i]
            sns.barplot(x='model', y=metric, data=results_df, ax=ax)
            ax.set_title(f'{name} Comparison')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('')
            ax.set_ylabel(name)
        
        # Remove empty subplot
        fig.delaxes(axes[-1])
        plt.tight_layout()
        plt.savefig('reports/figures/model_metrics_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2.2 Training Time Comparison
        if 'training_time' in results_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='model', y='training_time', data=results_df)
            plt.title('Model Training Time Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Training Time (seconds)')
            plt.xlabel('Model')
            plt.tight_layout()
            plt.savefig('reports/figures/training_time_comparison.png', bbox_inches='tight', dpi=300)
            plt.close()
        
    except Exception as e:
        print(f"Could not generate model comparison visualizations: {str(e)}")
    
    # 3. Generate correlation heatmap if data is available
    try:
        # Try to load the processed data
        processed_data_path = 'data/processed/processed_loan_data_encoded.csv'
        if os.path.exists(processed_data_path):
            df = pd.read_csv(processed_data_path)
            
            # Select only numeric columns for correlation
            numeric_df = df.select_dtypes(include=np.number)
            
            # Calculate correlation matrix
            corr = numeric_df.corr()
            
            # Plot the heatmap
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                       fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('reports/figures/feature_correlation.png', bbox_inches='tight', dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Could not generate correlation heatmap: {str(e)}")
    
    # 4. Generate class distribution visualization
    try:
        if 'default' in df.columns:
            plt.figure(figsize=(8, 6))
            class_dist = df['default'].value_counts(normalize=True) * 100
            plt.pie(class_dist, 
                   labels=['Not Default', 'Default'], 
                   autopct='%1.1f%%', 
                   startangle=90,
                   colors=['#66b3ff', '#ff9999'])
            plt.title('Class Distribution')
            plt.axis('equal')
            plt.savefig('reports/figures/class_distribution.png', bbox_inches='tight', dpi=300)
            plt.close()
            
    except Exception as e:
        print(f"Could not generate class distribution visualization: {str(e)}")
    
    plot_feature_importance()
    print("Visualizations generated in 'reports/figures/'")

if __name__ == "__main__":
    create_visualizations()
