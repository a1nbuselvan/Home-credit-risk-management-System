import pandas as pd
import matplotlib.pyplot as plt

def analyze_feature_importance(model, X, threshold=0.95):
    importance_scores = model.booster_.feature_importance(importance_type='gain')

    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    total_gain = feature_importance_df['importance'].sum()
    feature_importance_df['cumulative_importance'] = (
        feature_importance_df['importance'].cumsum() / total_gain
    )

    cutoff_index = feature_importance_df[
        feature_importance_df['cumulative_importance'] >= threshold
    ].index[0]
    
    relevant_features_df = feature_importance_df.iloc[:cutoff_index + 1].copy()
    num_features = len(relevant_features_df)

    print(f"\n--- FEATURES MANDATED BY {threshold*100}% RELEVANCE THRESHOLD ---")
    print(f"Total features originally: {len(X.columns)}")
    print(f"Features needed to reach {threshold*100}% total gain: {num_features}")
    print(relevant_features_df[['feature', 'importance', 'cumulative_importance']])

    plt.figure(figsize=(10, max(6, num_features * 0.3)))
    plt.barh(
        relevant_features_df['feature'][::-1], 
        relevant_features_df['importance'][::-1], 
        color='skyblue'
    )
    plt.xlabel('Total Gain Contribution')
    plt.title(f'All {num_features} Features Driving {threshold*100}% of Model Variance')
    plt.tight_layout()
    plt.show()

    return relevant_features_df