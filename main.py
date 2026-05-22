import loader
import aggregations
import model
import feature_importance

def main():
    ad_df, bur_df, burbal_df, poscash_df, ccbal_df, prevapp_df, install_df = loader.load_raw_data()

    print("Running aggregations...")
    bur_agg = aggregations.get_bureau_aggregations(bur_df)
    burbal_agg = aggregations.get_bureau_balance_aggregations(burbal_df, bur_df)
    pos_agg = aggregations.get_pos_cash_aggregations(poscash_df)
    cc_agg = aggregations.get_credit_card_aggregations(ccbal_df)
    prev_agg = aggregations.get_previous_application_aggregations(prevapp_df)
    install_agg = aggregations.get_installments_aggregations(install_df)

    print("Merging dataframes...")
    final_df = ad_df.copy()
    final_df = final_df.merge(bur_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(burbal_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(pos_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(cc_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(prev_agg, on="SK_ID_CURR", how="left")
    final_df = final_df.merge(install_agg, on="SK_ID_CURR", how="left")

    trained_model, X_payload = model.prepare_and_train(final_df)

    feature_importance_df = feature_importance.analyze_feature_importance(trained_model, X_payload, threshold=0.95)

    print("Retraining with selected features.")
    
    selected_features_list = feature_importance_df['feature'].tolist()
    
    final_model, final_payload = model.prepare_and_train(
        final_df, 
        features_to_use=selected_features_list
    )

if __name__ == "__main__":
    main()