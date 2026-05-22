import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

def prepare_and_train(final_df, features_to_use=None):
    if features_to_use is not None:
        X = final_df[features_to_use].copy()
    else:
        X = final_df.drop(columns=["SK_ID_CURR", "TARGET"])
        
    y = final_df["TARGET"]

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training LightGBM model with {X.shape[1]} features...")
    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        n_estimators=100,
        learning_rate=0.05,
        verbose=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    metrics = {"Accuracy": accuracy, "Precision": precision, "ROC-AUC": roc_auc}
    
    return model, X, X_test, y_test, metrics