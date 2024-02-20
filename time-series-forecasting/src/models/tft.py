import torch
import pandas as pd
import lightning.pytorch as pl
import wandb
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, QuantileLoss, TemporalFusionTransformer
from src.config import get_tft_config

tft_config = get_tft_config()


def create_tft_training_dataset(final_data):
    final_data['date'] = pd.to_datetime(final_data['date'])
    final_data['is_holiday'] = final_data['is_holiday'].astype(str)

    scaler = StandardScaler()
    features_to_normalize = ['temperature', 'radiation', 'wind_speed']
    scaler.fit(final_data[features_to_normalize])
    final_data[features_to_normalize] = scaler.transform(final_data[features_to_normalize])
    joblib.dump(scaler, f"{tft_config.get('output_path')}feature_scaler.joblib")

    train_data = final_data[final_data['date'] <= tft_config.get('training_cutoff_date')].copy()
    train_data.loc[:, 'time_idx'] = train_data['time_idx'].astype(int)

    target_normalizer = GroupNormalizer(groups=["country"], transformation="softplus")
    training_cutoff = train_data["time_idx"].max() - tft_config.get('max_prediction_length')

    training_dataset = TimeSeriesDataSet(
        train_data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="moer",
        group_ids=["country"],
        max_encoder_length=tft_config.get('max_encoder_length'),
        max_prediction_length=tft_config.get('max_prediction_length'),
        static_categoricals=["country"],
        time_varying_known_reals=["time_idx", "radiation", "wind_speed"],
        time_varying_unknown_reals=["moer"],
        target_normalizer=target_normalizer,
        lags={'moer': tft_config.get('lags')},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    training_dataset_params = training_dataset.get_parameters()
    torch.save(training_dataset_params, f"{tft_config.get('output_path')}/training_dataset_params.pth")
    return training_dataset


def create_tft_validation_dataset(final_data, training_dataset):
    validation_data = final_data[
        (final_data['date'] > tft_config.get('training_cutoff_date')) &
        (final_data['date'] <= tft_config.get('validation_cutoff_date'))
        ].copy()
    validation_data.loc[:, 'time_idx'] = validation_data['time_idx'].astype(int)

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        validation_data,
        predict=True,  # predict the decoder length on the last entries in the time index
        stop_randomization=True,
    )
    return validation_dataset


def create_tft_test_dataset(final_data, training_dataset):
    test_data = final_data[final_data['date'] > tft_config.get('validation_cutoff_date')].copy()
    test_data.loc[:, 'time_idx'] = test_data['time_idx'].astype(int)

    test_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        test_data,
        predict=True,  # predict the decoder length on the last entries in the time index
        stop_randomization=True,
    )
    return test_dataset


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
    torch.save(tft_model.state_dict(), f"{tft_config.get('output_path')}tft_model.pth")
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
    wandb_logger = WandbLogger(name="tsf_tft", project="tsf_tft")
    trainer = pl.Trainer(
        max_epochs=tft_config.get('max_epochs'),
        accelerator="auto",
        enable_model_summary=True,
        logger=wandb_logger,
        callbacks=[create_tft_checkpoints()]
    )
    return trainer


def train_tft(final_data):
    run = wandb.init(project="tsf_tft", config=dict(tft_config))

    final_data = final_data[final_data['country'] == 'DE'].copy()

    training_dataset = create_tft_training_dataset(final_data)
    validation_dataset = create_tft_validation_dataset(final_data, training_dataset)
    test_dataset = create_tft_test_dataset(final_data, training_dataset)

    train_dataloader = training_dataset.to_dataloader(train=True,
                                                      batch_size=tft_config.get('batch_size'),
                                                      num_workers=tft_config.get('num_workers'),
                                                      persistent_workers=True)
    val_dataloader = validation_dataset.to_dataloader(train=False,
                                                      batch_size=tft_config.get('batch_size') * 10,
                                                      num_workers=tft_config.get('num_workers'),
                                                      persistent_workers=True)
    test_dataloader = test_dataset.to_dataloader(train=False,
                                                 batch_size=tft_config.get('batch_size') * 10,
                                                 num_workers=tft_config.get('num_workers'),
                                                 persistent_workers=True)
    trainer = create_tft_trainer()
    tft_model = create_tft_model(training_dataset)

    trainer.fit(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(dataloaders=test_dataloader)

    model_save_path = f"{wandb.run.dir}/tft_model.pth"
    torch.save(tft_model.state_dict(), model_save_path)
    wandb.save(model_save_path)

    run.finish()
