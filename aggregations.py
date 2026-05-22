import pandas as pd

def get_bureau_aggregations(bur_df):
    bur_agg = bur_df.groupby("SK_ID_CURR").agg({
        "SK_ID_BUREAU": "count",
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"],
        "AMT_CREDIT_MAX_OVERDUE": ["max"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["mean"],
        "DAYS_ENDDATE_FACT": ["mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "AMT_ANNUITY": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"]
    })
    bur_agg.columns = ['BUR_' + '_'.join(col).upper() for col in bur_agg.columns]
    return bur_agg.reset_index()

def get_bureau_balance_aggregations(burbal_df, bur_df):
    burbal_merged = burbal_df.merge(
        bur_df[["SK_ID_BUREAU", "SK_ID_CURR"]],
        on="SK_ID_BUREAU",
        how="left"
    )
    status_dummies = pd.get_dummies(burbal_merged["STATUS"], prefix="STATUS")
    burbal_merged = pd.concat([burbal_merged, status_dummies], axis=1)

    burbal_agg = burbal_merged.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE": ["min", "max", "mean"],
        "STATUS_0": "sum",
        "STATUS_1": "sum",
        "STATUS_2": "sum",
        "STATUS_3": "sum",
        "STATUS_4": "sum",
        "STATUS_5": "sum",
        "STATUS_C": "sum",
        "STATUS_X": "sum",
    })
    burbal_agg.columns = ['BURBAL_' + '_'.join(col).upper() for col in burbal_agg.columns]
    return burbal_agg.reset_index()

def get_pos_cash_aggregations(poscash_df):
    pos_agg = poscash_df.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "nunique",
        "MONTHS_BALANCE": ["min", "max", "mean"],
        "CNT_INSTALMENT": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"]
    })
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    return pos_agg.reset_index()

def get_credit_card_aggregations(ccbal_df):
    cc_agg = ccbal_df.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "nunique",
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "AMT_DRAWINGS_CURRENT": ["sum", "mean"],
        "AMT_PAYMENT_CURRENT": ["sum", "mean"],
        "AMT_PAYMENT_TOTAL_CURRENT": ["sum"],
        "AMT_RECEIVABLE_PRINCIPAL": ["mean"],
        "AMT_TOTAL_RECEIVABLE": ["mean"],
        "CNT_DRAWINGS_CURRENT": ["sum"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "MONTHS_BALANCE": ["min", "max"]
    })
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    return cc_agg.reset_index()

def get_previous_application_aggregations(prevapp_df):
    prev_agg = prevapp_df.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "count",
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_ANNUITY": ["mean"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "AMT_GOODS_PRICE": ["mean"],
        "RATE_DOWN_PAYMENT": ["mean"],
        "CNT_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "SELLERPLACE_AREA": ["mean"],
        "NFLAG_INSURED_ON_APPROVAL": ["mean"]
    })
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    return prev_agg.reset_index()

def get_installments_aggregations(install_df):
    df = install_df.copy()
    df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["PAYMENT_DIFF"] = df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]

    install_agg = df.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": "nunique",
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "NUM_INSTALMENT_NUMBER": ["max"],
        "DAYS_INSTALMENT": ["min", "max"],
        "DAYS_ENTRY_PAYMENT": ["min", "max"],
        "AMT_INSTALMENT": ["sum", "mean"],
        "AMT_PAYMENT": ["sum", "mean"],
        "PAYMENT_DELAY": ["mean", "max"],
        "PAYMENT_DIFF": ["mean", "min"]
    })
    install_agg.columns = ['INST_' + '_'.join(col).upper() for col in install_agg.columns]
    return install_agg.reset_index()