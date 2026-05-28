"""
Trainer module untuk Bank Marketing Deposit Prediction Pipeline
rahadianivan09

FIXES vs versi sebelumnya:
- [FIX 1] steps_per_epoch=100 (was 50) → model lihat ~6400/8900 sampel per epoch
- [FIX 2] class_weight={0:1.0, 1:7.0} → handle imbalance ~88% no / 12% yes
- [FIX 3] Dropout setelah Dense ke-3 sebelum output layer
- [FIX 4] L2 regularization di semua Dense layers
- [FIX 5] hp extraction aman untuk dict maupun keras_tuner HyperParameters object
- [FIX 6] Tuner objective ganti ke val_auc (lebih robust untuk imbalanced data)
- [FIX 7] num_epochs=None di eval_dataset supaya tidak habis saat validation
"""

import os
from collections import namedtuple
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs


TunerFnResult = namedtuple('TunerFnResult', ['tuner', 'fit_kwargs'])

NUMERICAL_FEATURES = ['age', 'balance', 'day', 'campaign']
CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month'
]
LABEL_KEY = 'deposit'
EMBEDDING_DIM = 8

# Bank Marketing UCI: ~88% no (0), ~12% yes (1)
# class_weight supaya model tidak cheat ke majority class
CLASS_WEIGHT = {0: 1.0, 1: 7.0}

# Steps berdasarkan dataset size
# ~8900 train / 64 batch = ~139 steps/epoch → pakai 100 (aman, cepat)
# ~2200 eval  / 64 batch = ~34 steps       → pakai 25
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 25


def transformed_name(key):
    return key + '_xf'


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=64):
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def _get_hp(hp, key, default):
    """
    FIX 5: Safely extract HP value dari dict ATAU keras_tuner HyperParameters object.
    - Kalau hp=None       → pakai default
    - Kalau hp=dict       → dict.get(key, default)
    - Kalau hp=kt.HP obj  → hp.get(key), fallback ke default kalau KeyError
    """
    if hp is None:
        return default
    if isinstance(hp, dict):
        return hp.get(key, default)
    try:
        return hp.get(key)
    except (KeyError, AttributeError):
        return default


def build_model(hp=None):
    units_1 = _get_hp(hp, 'units_1', 128)
    units_2 = _get_hp(hp, 'units_2', 64)
    units_3 = _get_hp(hp, 'units_3', 32)
    dropout  = _get_hp(hp, 'dropout', 0.3)
    lr       = _get_hp(hp, 'learning_rate', 1e-3)

    # FIX 4: L2 regularizer untuk semua Dense
    l2 = tf.keras.regularizers.l2(1e-4)

    # Numerical inputs
    num_inputs, num_tensors = [], []
    for f in NUMERICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.float32)
        num_inputs.append(inp)
        num_tensors.append(inp)

    # Categorical inputs dengan embedding
    cat_inputs, cat_tensors = [], []
    for f in CATEGORICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=transformed_name(f), dtype=tf.int64)
        cat_inputs.append(inp)
        emb = tf.keras.layers.Embedding(input_dim=50, output_dim=EMBEDDING_DIM)(inp)
        emb = tf.keras.layers.Flatten()(emb)
        cat_tensors.append(emb)

    all_inputs  = num_inputs + cat_inputs
    all_tensors = num_tensors + cat_tensors
    x = tf.keras.layers.Concatenate()(all_tensors)

    # Layer 1
    x = tf.keras.layers.Dense(units_1, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Layer 2
    x = tf.keras.layers.Dense(units_2, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Layer 3 — FIX 3: tambah Dropout sebelum output
    x = tf.keras.layers.Dense(units_3, activation='relu', kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(dropout / 2)(x)  # dropout lebih kecil di layer terakhir

    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    model = tf.keras.Model(inputs=all_inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    model.summary()
    return model


def tuner_fn(fn_args: FnArgs):
    import keras_tuner as kt

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_epochs=5,
        batch_size=64
    )
    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        num_epochs=None,  # FIX 7: None supaya tidak habis saat validation
        batch_size=64
    )

    def build_model_for_tuner(hp):
        return build_model({
            'units_1': hp.Int('units_1', min_value=64, max_value=256, step=64),
            'units_2': hp.Int('units_2', min_value=32, max_value=128, step=32),
            'units_3': hp.Int('units_3', min_value=16, max_value=64, step=16),
            'dropout': hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1),
            'learning_rate': hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        })

    # FIX 6: objective=val_auc lebih robust untuk imbalanced data
    tuner = kt.RandomSearch(
        hypermodel=build_model_for_tuner,
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='bank_tuning'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'epochs': 5,
            'steps_per_epoch': STEPS_PER_EPOCH,
            'validation_steps': VALIDATION_STEPS,
            'class_weight': CLASS_WEIGHT,   # FIX 2
            'callbacks': [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=2,
                    mode='max'
                )
            ]
        }
    )


def get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_epochs=10,
        batch_size=64
    )
    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        num_epochs=None,  # FIX 7: None supaya tidak habis saat validation
        batch_size=64
    )

    # FIX 5: ekstraksi HP yang aman
    hp = fn_args.hyperparameters if fn_args.hyperparameters else None
    model = build_model(hp)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=2,
            mode='max',
            min_lr=1e-6
        ),
    ]

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=STEPS_PER_EPOCH,      # FIX 1: was 50, sekarang 100
        validation_data=eval_dataset,
        validation_steps=VALIDATION_STEPS,     # FIX 1: was 20, sekarang 25
        class_weight=CLASS_WEIGHT,             # FIX 2: handle imbalance
        callbacks=callbacks,
        verbose=1
    )

    signatures = {
        'serving_default': get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ),
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
    print(f'Model berhasil disimpan ke: {fn_args.serving_model_dir}')
