"""
Transform module untuk Bank Marketing Deposit Prediction Pipeline
rahadianivan09
"""

import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = ['age', 'balance', 'day', 'campaign']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']
LABEL_KEY = 'deposit'


def transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    outputs = {}

    # Label: konversi ke int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        tf.reshape(inputs[LABEL_KEY], [-1]), tf.int64
    )

    # Numerical features: normalisasi z-score
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            tf.cast(tf.reshape(inputs[feature], [-1]), tf.float32)
        )

    # Categorical features: vocabulary encoding
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            tf.reshape(inputs[feature], [-1]),
            vocab_filename=feature + '_vocab'
        )

    return outputs