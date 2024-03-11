training_dataset = TimeSeriesDataSet(
    train_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="moer",
    group_ids=["country"],
    max_encoder_length=24,
    max_prediction_length=24,
    static_categoricals=["country"],
    time_varying_known_categoricals=["season", "day_of_week", "is_holiday"],
    time_varying_known_reals=["ghi", "temperature", "wind_speed", "precipitation"],
    time_varying_unknown_reals=["moer"],
    target_normalizer=target_normalizer,
    lags={'moer': [24, 168]},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
