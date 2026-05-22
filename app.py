import streamlit as st
import pandas as pd
import io
import loader
import aggregations
import model
import feature_importance

st.set_page_config(page_title="Credit Risk Analyzer", layout="wide")

@st.cache_resource
def run_heavy_pipeline():
    ad_df, bur_df, burbal_df, poscash_df, ccbal_df, prevapp_df, install_df = loader.load_raw_data()
    
    bur_agg = aggregations.get_bureau_aggregations(bur_df)
    burbal_agg = aggregations.get_bureau_balance_aggregations(burbal_df, bur_df)
    pos_agg = aggregations.get_pos_cash_aggregations(poscash_df)
    cc_agg = aggregations.get_credit_card_aggregations(ccbal_df)
    prev_agg = aggregations.get_previous_application_aggregations(prevapp_df)
    install_agg = aggregations.get_installments_aggregations(install_df)
    
    final_df = ad_df.copy()
    final_df = final_df.merge(bur_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(burbal_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(pos_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(cc_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(prev_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(install_agg, on="SK_ID_CURR", how="left")
    
    trained_model, X_payload, X_test_base, y_test_base, base_metrics = model.prepare_and_train(final_df)
    
    importance_scores = trained_model.booster_.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': X_payload.columns,
        'importance': importance_scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    total_gain = feature_importance_df['importance'].sum()
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum() / total_gain
    
    cutoff_index = feature_importance_df[feature_importance_df['cumulative_importance'] >= 0.95].index[0]
    relevant_features_df = feature_importance_df.iloc[:cutoff_index + 1].copy()
    selected_features_list = relevant_features_df['feature'].tolist()
    
    final_model, final_payload, X_test_opt, y_test_opt, opt_metrics = model.prepare_and_train(
        final_df, 
        features_to_use=selected_features_list
    )
    
    base_numeric_cols = X_test_base.select_dtypes(include=['number']).columns.tolist()
    opt_numeric_cols = X_test_opt.select_dtypes(include=['number']).columns.tolist()
    
    safe_base_row = []
    risk_base_row = []
    for col in X_payload.columns:
        if col in base_numeric_cols:
            mean_val = float(X_test_base[col].mean())
            if pd.isna(mean_val):
                mean_val = 0.0
            safe_base_row.append(str(round(mean_val * 0.8, 2) if "DELAY" in col or "OVERDUE" in col or "DEBT" in col else round(mean_val * 1.2, 2)))
            risk_base_row.append(str(round(mean_val * 2.5, 2) if "DELAY" in col or "OVERDUE" in col or "DEBT" in col else round(mean_val * 0.5, 2)))
        else:
            safe_base_row.append("NaN")
            risk_base_row.append("NaN")
            
    safe_opt_row = []
    risk_opt_row = []
    for col in selected_features_list:
        if col in opt_numeric_cols:
            mean_val = float(X_test_opt[col].mean())
            if pd.isna(mean_val):
                mean_val = 0.0
            safe_opt_row.append(str(round(mean_val * 0.8, 2) if "DELAY" in col or "OVERDUE" in col or "DEBT" in col else round(mean_val * 1.2, 2)))
            risk_opt_row.append(str(round(mean_val * 2.5, 2) if "DELAY" in col or "OVERDUE" in col or "DEBT" in col else round(mean_val * 0.5, 2)))
        else:
            safe_opt_row.append("NaN")
            risk_opt_row.append("NaN")

    samples = {
        "all_features": {
            "safe": ", ".join(safe_base_row),
            "defaulter": ", ".join(risk_base_row)
        },
        "selected_features": {
            "safe": ", ".join(safe_opt_row),
            "defaulter": ", ".join(risk_opt_row)
        }
    }
    
    return trained_model, X_payload.columns.tolist(), base_metrics, final_model, selected_features_list, opt_metrics, relevant_features_df, samples

with st.spinner("Initializing Pipeline, Aggregating Data, and Training Models... Please wait."):
    trained_model, all_features, base_metrics, final_model, selected_features, opt_metrics, importance_df, samples = run_heavy_pipeline()

st.title("24GEP41 - Mini Project II - Credit Risk Prediction & Feature Optimization Engine")
st.markdown("24CSR013 AKASH K N || 24CSR022 ANBUSELVAN S || 24CSL370 MEGALA G")

tabs = st.tabs(["Feature Importance Insights", "Model Performance Comparison", "Live Risk Prediction"])

with tabs[0]:
    st.header("Model Intelligence & Feature Slicing")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Original Feature Count", value=len(all_features))
        st.metric(label="Optimized Feature Count (95% Gain)", value=len(selected_features))
        st.dataframe(importance_df, use_container_width=True)
    with col2:
        st.subheader("Dynamic Cumulative Gain Profile")
        st.bar_chart(data=importance_df.set_index('feature')['importance'].head(30))
        st.info("The chart above reflects the top driving factors behind borrower risk evaluations based on model information gain metrics.")

with tabs[1]:
    st.header("Baseline vs Optimized Model Comparison")
    
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "ROC-AUC"],
        "All Feature Model (Baseline)": [base_metrics["Accuracy"], base_metrics["Precision"], base_metrics["ROC-AUC"]],
        "Selected Feature Model (Optimized)": [opt_metrics["Accuracy"], opt_metrics["Precision"], opt_metrics["ROC-AUC"]]
    })
    
    st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Accuracy")
        st.bar_chart(comparison_df.set_index("Metric").loc[["Accuracy"]].T)
    with col2:
        st.subheader("Precision")
        st.bar_chart(comparison_df.set_index("Metric").loc[["Precision"]].T)
    with col3:
        st.subheader("ROC-AUC")
        st.bar_chart(comparison_df.set_index("Metric").loc[["ROC-AUC"]].T)

with tabs[2]:
    st.header("Real-Time Application Testing")
    
    model_choice = st.selectbox(
        "Choose Model Variant for Inference:",
        ["All Feature Model (Baseline)", "Selected Feature Model (Optimized - 50 Features)"]
    )
    
    active_model = trained_model if "All" in model_choice else final_model
    active_features = all_features if "All" in model_choice else selected_features
    sample_key = "all_features" if "All" in model_choice else "selected_features"
    
    st.subheader("Demonstration Shortcuts")
    col_btn1, col_btn2 = st.columns(2)
    
    if "text_input_value" not in st.session_state:
        st.session_state.text_input_value = ""
        
    with col_btn1:
        if st.button("Load Sample Safe Client Row"):
            st.session_state.text_input_value = samples[sample_key]["safe"]
            st.rerun()
            
    with col_btn2:
        if st.button("Load Sample Defaulter Client Row"):
            st.session_state.text_input_value = samples[sample_key]["defaulter"]
            st.rerun()

    st.markdown(f"**Expected CSV Header Input Format ({len(active_features)} values needed):**")
    st.text_area("Copy these headers to format your inputs:", value=", ".join(active_features), height=70, disabled=True)
    
    csv_input = st.text_area(
        "Paste a single row of comma-separated values matching the schema above:",
        value=st.session_state.text_input_value,
        placeholder="e.g., 0.45, 120000, -14000, ..."
    )
    
    if st.button("Evaluate Default Risk Probability"):
        if csv_input:
            try:
                raw_values = [v.strip() for v in csv_input.split(",")]
                
                if len(raw_values) != len(active_features):
                    st.error(f"Input mismatch! Expected {len(active_features)} variables, but received {len(raw_values)} parameters.")
                else:
                    processed_values = []
                    for val in raw_values:
                        if val.lower() == "nan" or val == "":
                            processed_values.append(None)
                        else:
                            try:
                                processed_values.append(float(val))
                            except ValueError:
                                processed_values.append(val)
                                
                    input_df = pd.DataFrame([processed_values], columns=active_features)
                    
                    for col in input_df.columns:
                        if input_df[col].dtype == 'object':
                            input_df[col] = input_df[col].astype('category')
                        else:
                            input_df[col] = pd.to_numeric(input_df[col])
                            
                    probability = active_model.predict_proba(input_df)[:, 1][0]
                    risk_percentage = probability * 100
                    
                    st.markdown("---")
                    st.subheader("Analysis Verdict")
                    
                    if risk_percentage > 15:
                        st.error(f"High Default Risk Detected: This profile has a {risk_percentage:.2f}% likelihood of defaulting.")
                    elif risk_percentage > 7:
                        st.warning(f"Medium Risk Warning: This profile has a {risk_percentage:.2f}% likelihood of defaulting.")
                    else:
                        st.success(f"Low Risk / Safe Profile: This profile has a {risk_percentage:.2f}% likelihood of defaulting.")
                        
            except Exception as e:
                st.error(f"An error occurred parsing inputs: {str(e)}")
        else:
            st.info("Please paste raw comma-separated records into the text box above to generate risk percentages.")