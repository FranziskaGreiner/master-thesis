training_dataset = TimeSeriesDataSet(
    train_data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="moer",
    group_ids=["country"],
    max_encoder_length=168,
    max_prediction_length=168,
    static_categoricals=["country"],
    time_varying_known_categoricals=["is_holiday_or_weekend"],
    time_varying_known_reals=["time_idx", "ghi", "temperature", "wind_speed", "precipitation", "season_sin", "season_cos", "day_of_week_sin", "day_of_week_cos", "day_of_year_sin", "day_of_year_cos", "hour_of_day_sin", "hour_of_day_cos"],
    time_varying_unknown_reals=["moer"],
    target_normalizer=target_normalizer,
    lags={'moer': [168]},
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
