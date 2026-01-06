import pandas as pd 
import numpy as np 
import config, common_utils, data_prep, feature_creation, model_building
import os 

def build_and_score_models(final_features = None):
    #-------Pre process data and aggregate to daily level -------------------
    try:
        df_daily = pd.read_csv(config.dalily_agg_filepath)
    except:
        df_daily = data_prep.load_and_prepare_daily_meter_data(config.FILE_PATH, config.DELIM)
    #--------Create features and get final model data ------------------------------------
    try:
        model_data = pd.read_parquet(config.model_data_filepath)
    except:
        model_data = feature_creation.get_model_data(df_daily)
    #---- Get list of final features post funnel-------------------------
    if(final_features==None):
        final_features, filtered_features, stats_df , model_data= model_building.feature_funnel_selector(
        model_data=model_data,
        target_col="daily_consumption",
        feature_cols=None,   # pass your list; or leave None to auto-pick numeric
        data_flag_col="data_flag",
        train_value="train"
        # sample_n=200_000
    )
    # ----  Train models + get metrics/importances ----
    df_with_preds, metrics_df, importance_df, models= model_building.train_models_log_target_with_importance(
        df=model_data,
        feature_cols= final_features,
        target_col="daily_consumption",
        data_flag_col="data_flag",
        train_flag="train",
        test_flag="test",
        pred_prefix="pred",
    )
    metrices_df = metrics_df.sort_values(["model", "split"])
    model_building.save_models(models)
    common_utils.save_xls([df_with_preds[df_with_preds.data_flag=='test'], metrics_df, importance_df], os.path.join(config.res_dir, "model_results_all.xlsx"))
    return df_with_preds, models, metrices_df