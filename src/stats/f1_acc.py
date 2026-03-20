from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import os

def generate_adaptation_metrics(data, model_name, output_path):
    # or1_full, etc., from your accumulated overall_runs_full
    # We need to collect ALL predicted vs ALL targets
    
    all_targets = []
    all_preds = []
    
    # 1. Collect data from the nested dictionary
    for topic, runs in data.items():
        for run_id, run_data in runs.items():
            # Assuming 'full' contains a list of dicts with 'formality' (target) 
            # and 'predicted_formality' (detected by your classifier)
            samples = run_data.get('full', [])
            all_targets.extend([s.get('input')[0] for s in samples])
            all_preds.extend([s.get('output')[0] for s in samples])

    # 2. Calculate F1 Scores
    # 'weighted' accounts for label imbalance; 'macro' treats F1, F2, F3 equally
    f1_overall = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, output_dict=True)

    # 3. Adaptation vs Non-Adaptation Table
    adaptation_data = []
    for f in [1, 2, 3]:
        f_str = str(f)
        if f_str in report:
            acc = report[f_str]['recall'] * 100 # Recall for the class is the adaptation %
            adaptation_data.append({
                "Formality": f"F{f}",
                "Adapted %": f"{acc:.2f}%",
                "Non-Adapted %": f"{100 - acc:.2f}%",
                "F1 Score": f"{report[f_str]['f1-score']:.2f}"
            })

    # Add Overall row
    adaptation_data.append({
        "Formality": "OVERALL",
        "Adapted %": f"{report['accuracy']*100:.2f}%",
        "Non-Adapted %": f"{(1-report['accuracy'])*100:.2f}%",
        "F1 Score": f"{f1_overall:.2f}"
    })

    # Save to CSV/TXT
    df = pd.DataFrame(adaptation_data)
    df.to_csv(os.path.join(output_path, f"{model_name}_formality_adaptation.csv"), index=False)
    print(df)