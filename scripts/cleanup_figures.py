import os
import shutil
from pathlib import Path

def clean_figures():
    # Define directories
    figures_dir = Path('reports/figures')
    
    # List of files to keep
    files_to_keep = {
        'model_metrics_comparison.png',
        'training_time_comparison.png',
        'model_comparison.png',
        'class_distribution.png',
        'feature_importance.png',
        'feature_correlation.png',  # New correlation heatmap
        'demographic_default_rates.png',
        'risk_segments.png'
    }
    
    # Create a backup directory for old figures
    backup_dir = figures_dir / 'archive'
    backup_dir.mkdir(exist_ok=True, parents=True)
    
    # Counters
    kept_count = 0
    moved_count = 0
    
    # Process each file in the figures directory
    for item in figures_dir.iterdir():
        if item.is_file():
            if item.name in files_to_keep:
                kept_count += 1
                print(f"Keeping: {item.name}")
            else:
                # Move to archive
                try:
                    shutil.move(
                        str(item),
                        str(backup_dir / item.name)
                    )
                    moved_count += 1
                    print(f"Moved to archive: {item.name}")
                except Exception as e:
                    print(f"Error moving {item.name}: {str(e)}")
    
    # Print summary
    print("\nCleanup complete!")
    print(f"Kept {kept_count} files in {figures_dir}")
    print(f"Moved {moved_count} files to {backup_dir}")

if __name__ == "__main__":
    clean_figures()
