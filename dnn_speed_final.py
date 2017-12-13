import os
import collections

import tensorflow as tf

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"


# Decode a line from the CSV.
def csv_decoder(line):
    """Convert a CSV row to a dictonary of features."""
    parsed = tf.decode_csv(line, list(csv_defaults.values()))
    return dict(zip(csv_defaults.keys(), parsed))


# The train file has an extra empty line at the end.
# We'll use this method to filter that out.
def filter_empty_lines(line):
    return tf.not_equal(tf.size(tf.string_split([line], ',').values), 0)


def create_train_input_fn(path, label_name='label',
                          repeat_count=100, shuffle_buffer=1024*4, batch_size=32):
    def input_fn():
        dataset = (
            tf.data.TextLineDataset(path)  # create a dataset from a file
                .filter(filter_empty_lines)  # ignore empty lines
                .map(csv_decoder)  # parse each row
                .shuffle(buffer_size=shuffle_buffer)  # shuffle the dataset
                .repeat(repeat_count)  # repeate indefinitely
                .batch(batch_size))  # batch the data

        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()
        labels = columns.pop(label_name)

        return columns, labels

    return input_fn


def create_test_input_fn(path, label_name='label'):
    def input_fn():
        dataset = (
            tf.data.TextLineDataset(path)
                # .skip(1)  # The test file has a strange first line, we want to ignore this.
                .filter(filter_empty_lines)
                .map(csv_decoder)
                .batch(32))

        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()

        # separate the label and convert it to true/false
        labels = columns.pop(label_name)
        return columns, labels
    return input_fn


if __name__ == '__main__':
    # Fetch and store Training and Test dataset files
    PATH = "tf_dataset_and_estimator_apis"

    PATH_DATASET = "dataset"
    TRAIN_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_training_v_speed1212.csv'
    TEST_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test_fspeed1212.csv'

    SERVABLE_MODEL_DIR = "serving_savemodel"

    tf.logging.set_verbosity(tf.logging.INFO)

    csv_defaults = collections.OrderedDict(
        [('status', [0.0]),
        ('location_speed', [0.0]),
        ('httpcode_5m_200', [0.0]),
        ('httpcode_5m_302', [0.0]),
        ('httpcode_5m_404', [0.0]),
        ('httpcode_5m_403', [0.0]),
        ('httpcode_5m_500', [0.0]),
        ('httpcode_30m_200', [0.0]),
        ('httpcode_30m_302', [0.0]),
        ('httpcode_30m_404', [0.0]),
        ('httpcode_30m_403', [0.0]),
        ('httpcode_30m_500', [0.0]),
        ('httpcode_1d_200', [0.0]),
        ('httpcode_1d_302', [0.0]),
        ('httpcode_1d_404', [0.0]),
        ('httpcode_1d_403', [0.0]),
        ('httpcode_1d_500', [0.0]),
        ('body_bytes_sent', [0.0]),
        ('upstream_response_time', [0.0]),
        ('httpcode_total_200', [0.0]),
        ('httpcode_total_302', [0.0]),
        ('httpcode_total_404', [0.0]),
        ('httpcode_total_403', [0.0]),
        ('httpcode_total_500', [0.0]),
        ('request_time', [0.0]),
        ('remote_addr', ['']),
        ('location_city', ['']),
        ('label', [0])]
    )

    # Feature columns describe how to use the input.
    active_feature_column = [
                     'status',
                     'location_speed',
                     'httpcode_5m_200',
                     'httpcode_5m_302',
                     'httpcode_5m_404',
                     'httpcode_5m_403',
                     'httpcode_5m_500',
                     'httpcode_30m_200',
                     'httpcode_30m_302',
                     'httpcode_30m_404',
                     'httpcode_30m_403',
                     'httpcode_30m_500',
                     'httpcode_1d_200',
                     'httpcode_1d_302',
                     'httpcode_1d_404',
                     'httpcode_1d_403',
                     'httpcode_1d_500',
                     'body_bytes_sent',
                     'upstream_response_time',
                     'httpcode_total_200',
                     'httpcode_total_302',
                     'httpcode_total_404',
                     'httpcode_total_403',
                     'httpcode_total_500',
                     'request_time',
                     'remote_addr',
                     'location_city'
                    ]

    # Create the feature_columns, which specifies the input to our model
    feature_columns = []
    for key in active_feature_column[:-2]:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('remote_addr', 100), dimension=5))
    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('location_city', 100), dimension=5))

    # Create a deep neural network regression classifier
    # Use the DNNClassifier pre-made estimator
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[800, 400, 200,100,50],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.01
        ),
        n_classes=2,
        model_dir=PATH)  # Path to where checkpoints etc are stored

    train_input_fn = create_train_input_fn(TRAIN_URL, label_name='label', repeat_count=100)
    test_input_fn = create_train_input_fn(TEST_URL, label_name='label')

    classifier.train(
        input_fn=train_input_fn )

    # Evaluate our model using the examples contained in FILE_TEST
    # Return value will contain evaluation_metrics such as: loss & average_loss
    evaluate_result = classifier.evaluate(
        input_fn=test_input_fn )

    print("Evaluation results")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))

    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    servable_model_path = classifier.export_savedmodel(SERVABLE_MODEL_DIR, export_input_fn)