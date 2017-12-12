
from __future__ import print_function, division

import os
import collections
from pandas import Series, DataFrame
import tensorflow as tf
from tensorflow.python.framework import ops

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
# from sklearn.cross_validation import train_test_split
import random

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

    PATH_DATASET = "~/TensorFlow-Examples/1_Introduction"
    TRAIN_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_training_v_speed1212.csv'
    # TEST_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test3.csv'
    TESTo_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test_v_speed1212.csv'
    MS_URL='/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/mean_std_v_speed1212.csv'
    MS=pd.read_csv(MS_URL)
    TEST = pd.read_csv(TESTo_URL)
    mup=MS['mean']
    stdp=MS['std']
    # print(mup)
    # print(stdp)

    listdatao = [
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
        'request_time']

    sortlist3 = [
        'remote_addr',
        'location_city',
        'label']

    datao=TEST.reindex(columns=listdatao)
    datao2=TEST.reindex(columns=sortlist3)
    # print(TEST[listdatao[0]][0])
    # print(type(datao))
    fs1, fs2 = datao.shape
    # print(fs1)
    # print(fs2)
    # print(meanp)
    # print(stdp)
    # print(frameanp2.shape)
    fdatao = np.array(datao)
    # print(fdatao)
    fdatao2 =fdatao
    # F_scaled2 = np.zeros((fs1, fs2))
    # # print(F_scaled3[1][1])
    for i in range(0, fs1):
        for j in range(0, fs2):
            fdatao2[i][j] = (fdatao[i][j] - mup[j]) / stdp[j]


    # print(fdatao2)

    sfdatao2=DataFrame(fdatao2)

    alldata = [sfdatao2, datao2]
    sff = pd.concat(alldata, axis=1)

    pd.DataFrame.to_csv(sff, '~/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test_fspeed1212.csv', encoding='utf8',header=None,
                        index=None)
    TEST_URL = '/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test_fspeed1212.csv'
    # PATH = "tf_dataset_and_estimator_apis"
    #
    # PATH_DATASET = "dataset"
    # TRAIN_URL = PATH_DATASET + os.sep + "security_test.csv"
    # TEST_URL = PATH_DATASET + os.sep + "security_test.csv"

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
        feature_columns=feature_columns,  # The input features to our model
        hidden_units=[100,50],  # Two layers, each with 10 neurons
        optimizer=tf.train.AdamOptimizer(
          learning_rate=0.0001
        ),
        n_classes=2,
        model_dir=PATH)  # Path to where checkpoints etc are stored

    train_input_fn = create_train_input_fn(TRAIN_URL, label_name='label', repeat_count=10)
    test_input_fn = create_train_input_fn(TEST_URL, label_name='label')

    # with tf.Session() as sess:
    #         print sess.run(test_input_fn)

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
    print(evaluate_result)

    # {'loss': 11.98338, 'accuracy_baseline': 0.73939395, 'global_step': 929, 'auc': 0.92053467, 'prediction/mean': 0.81986403, 'label/mean': 0.73939395, 'average_loss': 0.374753, 'auc_precision_recall': 0.97038209, 'accuracy': 0.80000001}

    #classifier.export_savedmodel(MODEL_PATH, serving_input_receiver_fn=serving_input_receiver_fn)

    #import os
    #os.system('rm -r /Users/wdai/work/ml/DNN/tf_dataset_and_estimator_apis/*')