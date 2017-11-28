
import os
import collections

import tensorflow as tf
from tensorflow.python.framework import ops

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

<<<<<<< HEAD
# Fetch and store Training and Test dataset files
PATH = "tf_dataset_and_estimator_apis"
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "security_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "security_test.csv"

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training & test data
feature_names = ['rank',
                 'status',
                 'location_speed',
                 'lremote_addr',
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
                 'location_city',
                 'is_good'
]

# make sure we didn't count column wrong
assert feature_names[-2] == 'location_city'
assert feature_names[-1] == 'is_good'

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        feature_type = [[0.0]] * (len(feature_names) - 2)
        feature_type.append([''])
        feature_type.append([0])
        feature_type[3] = [''] # ip address
        parsed_line = tf.decode_csv(line, feature_type)
        label = parsed_line[-1:]
        del parsed_line[-1]  # Delete label
        del parsed_line[0]   # Delete rank
        features = parsed_line  # Everything but label are the features
        d = dict(zip(feature_names[1:-1], features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = my_input_fn(FILE_TRAIN, True)  # Will return 32 random elements

# Create the feature_columns, which specifies the input to our model
# All our input features are numeric, so use numeric_column for each one
# Pay attention!!!!! we need change here if we modified the features
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names[1:2]]
feature_columns.append(
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('lremote_addr', 100), dimension=5))
feature_columns += [tf.feature_column.numeric_column(k) for k in feature_names[4:-2]]
feature_columns.append(
    tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_hash_bucket('location_city', 100), dimension=5))

# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[20, 10],  # Two layers, each with 10 neurons
    n_classes=2,
    model_dir=PATH)  # Path to where checkpoints etc are stored

# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 8 iterations of train data (epochs)
classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 4))
print("Evaluation results")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))
=======
>>>>>>> e00e32284061baf4283ed2887b408d1eddcfd37d

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
    TRAIN_URL = PATH_DATASET + os.sep + "security_training.csv"
    TEST_URL = PATH_DATASET + os.sep + "security_test.csv"

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
        ('lremote_addr', ['']),
        ('location_city', ['']),
        ('is_good', [0])]
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
                     'lremote_addr',
                     'location_city'
                    ]

    # Create the feature_columns, which specifies the input to our model
    feature_columns = []
    for key in active_feature_column[:-2]:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('lremote_addr', 100), dimension=5))
    feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket('location_city', 100), dimension=5))

    # Create a deep neural network regression classifier
    # Use the DNNClassifier pre-made estimator
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,  # The input features to our model
        hidden_units=[100, 50, 10],  # Two layers, each with 10 neurons
        optimizer=tf.train.AdamOptimizer(
          learning_rate=0.01
        ),
        n_classes=2,
        model_dir=PATH)  # Path to where checkpoints etc are stored

    train_input_fn = create_train_input_fn(TRAIN_URL, label_name='is_good', repeat_count=100)
    test_input_fn = create_train_input_fn(TEST_URL, label_name='is_good')
    # Train our model, use the previously function my_input_fn
    # Input to training is a file with training example
    # Stop training after 8 iterations of train data (epochs)
    classifier.train(
        input_fn=train_input_fn )

    # Evaluate our model using the examples contained in FILE_TEST
    # Return value will contain evaluation_metrics such as: loss & average_loss
    evaluate_result = classifier.evaluate(
        input_fn=test_input_fn )

    print("Evaluation results")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))

    #classifier.export_savedmodel(MODEL_PATH, serving_input_receiver_fn=serving_input_receiver_fn)

    #import os
    #os.system('rm -r /Users/wdai/work/ml/DNN/tf_dataset_and_estimator_apis/*')