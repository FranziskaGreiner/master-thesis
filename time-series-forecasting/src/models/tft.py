import torch
import pandas as pd
import lightning.pytorch as pl
import wandb
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss, TemporalFusionTransformer, Baseline
from pytorch_forecasting.metrics import MAE
from src.config import get_tft_config
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

tft_config = get_tft_config()


def add_time_idx(weather_time_moer_data):
    weather_time_moer_data.loc[:, 'date'] = pd.to_datetime(weather_time_moer_data['date'])
    min_date_per_group = weather_time_moer_data.groupby('country')['date'].transform('min')
    weather_time_moer_data.loc[:, 'time_idx'] = ((weather_time_moer_data['date'] - min_date_per_group)
                                                 .dt.total_seconds() / 3600).astype(int)
    return weather_time_moer_data


def normalize_features(weather_time_moer_data):
    scaler = StandardScaler()
    features_to_normalize = ['temperature', 'ghi', 'wind_speed', 'precipitation']
    scaler.fit(weather_time_moer_data[features_to_normalize])
    weather_time_moer_data.loc[:, features_to_normalize] = scaler.transform(
        weather_time_moer_data[features_to_normalize])

    scaler_save_path = f"{wandb.run.dir}/feature_scaler.joblib"
    joblib.dump(scaler, scaler_save_path)
    return weather_time_moer_data


def convert_categoricals(weather_time_moer_data):
    weather_time_moer_data.loc[:, 'season'] = weather_time_moer_data['season'].astype(str)
    weather_time_moer_data.loc[:, 'day_of_week'] = weather_time_moer_data['day_of_week'].astype(str)
    weather_time_moer_data.loc[:, 'is_holiday'] = weather_time_moer_data['is_holiday'].astype(str)
    return weather_time_moer_data


def create_training_validation_data(weather_time_moer_data):
    cutoffs = {}
    for country in ['DE', 'NO']:
        country_data = weather_time_moer_data[weather_time_moer_data['country'] == country]
        max_time_idx = country_data["time_idx"].max()
        cutoffs[country] = max_time_idx - tft_config.get('max_prediction_length')

    train_data = pd.DataFrame()
    validation_data = pd.DataFrame()

    for country, cutoff in cutoffs.items():
        country_train_data = weather_time_moer_data[(weather_time_moer_data['country'] == country) &
                                                    (weather_time_moer_data['time_idx'] <= cutoff)]
        country_validation_data = weather_time_moer_data[(weather_time_moer_data['country'] == country) &
                                                         (weather_time_moer_data['time_idx'] > cutoff)]
        train_data = pd.concat([train_data, country_train_data])
        validation_data = pd.concat([validation_data, country_validation_data])

    print(train_data.tail())
    print(validation_data.tail())
    return train_data, validation_data


def create_tft_training_dataset(train_data):
    target_normalizer = GroupNormalizer(groups=["country"], transformation="softplus")

    training_dataset = TimeSeriesDataSet(
        train_data,
        time_idx=tft_config.get("time_idx"),
        target=tft_config.get("target"),
        group_ids=tft_config.get("group_ids"),
        max_encoder_length=tft_config.get('max_encoder_length'),  # lookback window
        max_prediction_length=tft_config.get('max_prediction_length'),
        static_categoricals=tft_config.get("static_categoricals"),
        time_varying_known_categoricals=tft_config.get("time_varying_known_categoricals"),
        time_varying_known_reals=tft_config.get("time_varying_known_reals"),
        time_varying_unknown_reals=tft_config.get("time_varying_unknown_reals"),
        target_normalizer=target_normalizer,
        lags=tft_config.get("lags"),
        add_relative_time_idx=tft_config.get("add_relative_time_idx"),
        add_target_scales=tft_config.get("add_target_scales"),
        add_encoder_length=tft_config.get("add_encoder_length"),
        allow_missing_timesteps=tft_config.get("allow_missing_timesteps")
    )
    training_dataset_params = training_dataset.get_parameters()
    training_params_save_path = f"{wandb.run.dir}/training_dataset_params.pth"
    torch.save(training_dataset_params, training_params_save_path)
    return training_dataset


def create_tft_validation_dataset(training_dataset, validation_data):
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_data,
        predict=True,  # predict the decoder length on the last entries in the time index
        stop_randomization=True,
    )
    return validation_dataset


def create_tft_test_dataset(weather_time_moer_data, training_dataset):
    test_data = weather_time_moer_data[weather_time_moer_data['date'] > tft_config.get('validation_cutoff_date')].copy()

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_data,
        predict=True,  # predict the decoder length on the last entries in the time index
        stop_randomization=True,
    )
    return test_dataset


def create_baseline_model(val_dataloader):
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    mae_value = MAE()(baseline_predictions.output, baseline_predictions.y)
    print(f"Baseline MAE: {mae_value.item()}")


def create_tft_model(training_dataset):
    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=tft_config.get('learning_rate'),
        hidden_size=tft_config.get('hidden_size'),
        attention_head_size=tft_config.get('attention_head_size'),
        dropout=tft_config.get('dropout'),
        hidden_continuous_size=tft_config.get('hidden_continuous_size'),
        output_size=tft_config.get('output_size'),  # 7 quantiles by default
        loss=QuantileLoss(),
        reduce_on_plateau_patience=tft_config.get('reduce_on_plateau_patience'),  # reduce learning automatically
    )
    model_save_path = Path(wandb.run.dir) / "tft_model.pth"
    torch.save(tft_model.state_dict(), model_save_path)
    return tft_model


def create_tft_checkpoints():
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=Path(wandb.run.dir) / "checkpoints",
        filename='tft-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    return checkpoint_callback


def create_tft_trainer():
    wandb_logger = WandbLogger(name="tsf_moer_tft", project="tsf_moer_tft")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=tft_config.get('max_epochs'),
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=tft_config.get('gradient_clip_val'),
        logger=wandb_logger,
        callbacks=[create_tft_checkpoints(), early_stop_callback]
    )
    return trainer


def train_tft(weather_time_moer_data):
    run = wandb.init(project="tsf_moer__tft", config=dict(tft_config))

    weather_time_moer_data = add_time_idx(weather_time_moer_data)
    weather_time_moer_data = normalize_features(weather_time_moer_data)
    weather_time_moer_data = convert_categoricals(weather_time_moer_data)

    train_data, validation_data = create_training_validation_data(weather_time_moer_data)

    training_dataset = create_tft_training_dataset(train_data)
    validation_dataset = create_tft_validation_dataset(training_dataset, validation_data)

    train_dataloader = training_dataset.to_dataloader(train=True,
                                                      batch_size=tft_config.get('batch_size'),
                                                      num_workers=tft_config.get('num_workers'),
                                                      persistent_workers=True)
    val_dataloader = validation_dataset.to_dataloader(train=False,
                                                      batch_size=tft_config.get('batch_size') * 10,
                                                      num_workers=tft_config.get('num_workers'),
                                                      persistent_workers=True)

    create_baseline_model(val_dataloader)
    trainer = create_tft_trainer()
    tft_model = create_tft_model(training_dataset)

    # find optimal learning rate
    res = Tuner(trainer).lr_find(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(suggest=True, show=False)
    lr_plot_file_path = f"{wandb.run.dir}/learning_rate.png"
    fig.savefig(lr_plot_file_path)
    plt.show()
    plt.close(fig)

    trainer.fit(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # hyperparameter tuning
    # study = optimize_hyperparameters(
    #     train_dataloader,
    #     val_dataloader,
    #     model_path="optuna_test",
    #     n_trials=50,
    #     max_epochs=10,
    #     gradient_clip_val_range=(0.01, 1.0),
    #     hidden_size_range=(8, 128),
    #     hidden_continuous_size_range=(8, 128),
    #     attention_head_size_range=(1, 4),
    #     learning_rate_range=(0.001, 0.1),
    #     dropout_range=(0.1, 0.4),
    #     trainer_kwargs=dict(limit_train_batches=30),
    #     reduce_on_plateau_patience=4,
    #     use_learning_rate_finder=False,
    # )
    #
    # hyperparameter_study_save_path = f"{wandb.run.dir}/hyperparameter_study.pkl"
    # joblib.dump(study, hyperparameter_study_save_path)
    # print(study.best_trial.params)

    # trainer.test(dataloaders=test_dataloader, ckpt_path='best')

    model_save_path = f"{wandb.run.dir}/tft_model.pth"
    torch.save(tft_model.state_dict(), model_save_path)
    wandb.save(model_save_path)

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    val_prediction_results = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    for idx in range(2):
        fig, ax = plt.subplots(figsize=(23, 5))
        best_tft.plot_prediction(val_prediction_results.x,
                                 val_prediction_results.output,
                                 idx=idx,
                                 add_loss_to_title=True,
                                 ax=ax)
        val_plot_file_path = f"{wandb.run.dir}/plot_val_group_{idx}.png"
        plt.savefig(val_plot_file_path)
        wandb.log({f"plot_val_group_{idx}": wandb.Image(val_plot_file_path)})
        plt.show()
        plt.close()

    run.finish()
