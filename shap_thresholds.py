import numpy as np
import pandas as pd
import shap

def find_positive_shap_thresholds(personas_shaps, preprocessor=None, persona=None, top_n=10, min_samples=10, rolling_window=20):
    """
    Identify features with distinguishable positive SHAP values and estimate value thresholds.
    """

    if persona:
        data = personas_shaps[persona]
        shap_vals = data['shap_values'][:, :, 1]
        X_transformed = data['X_transformed']
        sample_name = persona
    else:
        all_shap_vals = []
        all_X = []
        for p in personas_shaps.values():
            all_shap_vals.append(p['shap_values'][:, :, 1])
            all_X.append(p['X_transformed'])
        shap_vals = np.vstack(all_shap_vals)
        X_transformed = np.vstack(all_X) if not isinstance(all_X[0], pd.DataFrame) else pd.concat(all_X, ignore_index=True)
        sample_name = 'ALL'

    # Handle feature names with or without preprocessor
    if preprocessor is not None:
        feature_names = preprocessor.get_feature_names_out()
    elif isinstance(X_transformed, pd.DataFrame):
        feature_names = X_transformed.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

    if shap_vals.shape[0] != (X_transformed.shape[0] if isinstance(X_transformed, np.ndarray) else X_transformed.shape[0]):
        raise ValueError(f"SHAP values shape {shap_vals.shape} does not match transformed data shape {X_transformed.shape}.")

    # Convert to DataFrame for consistent behavior
    X_df = pd.DataFrame(X_transformed, columns=feature_names) if not isinstance(X_transformed, pd.DataFrame) else X_transformed.copy()

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    results = []
    print(f"\n=== Top {top_n} Features for: {sample_name} ===")

    for idx in top_indices:
        fname = feature_names[idx]
        f_vals = X_df[fname].values
        s_vals = shap_vals[:, idx]

        df = pd.DataFrame({'feature_val': f_vals, 'shap_val': s_vals}).sort_values('feature_val')
        df['shap_rolling'] = df['shap_val'].rolling(window=rolling_window, min_periods=1).mean()

        pos_thresh = df[df['shap_rolling'] > 0]

        if len(pos_thresh) >= min_samples:
            threshold = pos_thresh['feature_val'].iloc[0]
            if pd.api.types.is_bool_dtype(pos_thresh['feature_val']):
                results.append((fname, bool(threshold)))
            else:
                results.append((fname, round(threshold, 2)))
            print(f"• {fname}: SHAP > 0 at ~ {threshold}")

    if not results:
        print("No clearly distinguishable positive thresholds found.")

    return results
