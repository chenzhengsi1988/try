
#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with security_dnn model.

The client downloads test data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000 --concurrentcy 10
"""

from __future__ import print_function

import sys
import threading
import time
import csv

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_integer('concurrency', 100,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 10000, 'Number of test images')
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('model', 'DNN', 'Model name.')
FLAGS = tf.app.flags.FLAGS
DATA_FILE = 'dataset/security_training.csv'

class _ResultCounter(object):
    """Counter for the prediction results."""

    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
        return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def data_reader(data_file):
    with open(data_file, 'rb') as f:
        reader = unicode_csv_reader(f)
        data_list = list(reader)

    feature_dict_list = []
    for row in data_list:
        row = [float(item) for item in row[:-3]] + row[-3:]

        feature_dict = {
            'status': _float_feature(value=row[0]),
            'location_speed': _float_feature(value=row[1]),
            'httpcode_5m_200': _float_feature(value=row[2]),
            'httpcode_5m_302': _float_feature(value=row[3]),
            'httpcode_5m_404': _float_feature(value=row[4]),
            'httpcode_5m_403': _float_feature(value=row[5]),
            'httpcode_5m_500': _float_feature(value=row[6]),
            'httpcode_30m_200': _float_feature(value=row[7]),
            'httpcode_30m_302': _float_feature(value=row[8]),
            'httpcode_30m_404': _float_feature(value=row[9]),
            'httpcode_30m_403': _float_feature(value=row[10]),
            'httpcode_30m_500': _float_feature(value=row[11]),
            'httpcode_1d_200': _float_feature(value=row[12]),
            'httpcode_1d_302': _float_feature(value=row[13]),
            'httpcode_1d_404': _float_feature(value=row[14]),
            'httpcode_1d_403': _float_feature(value=row[15]),
            'httpcode_1d_500': _float_feature(value=row[16]),
            'body_bytes_sent': _float_feature(value=row[17]),
            'upstream_response_time': _float_feature(value=row[18]),
            'httpcode_total_200': _float_feature(value=row[19]),
            'httpcode_total_302': _float_feature(value=row[20]),
            'httpcode_total_404': _float_feature(value=row[21]),
            'httpcode_total_403': _float_feature(value=row[22]),
            'httpcode_total_500': _float_feature(value=row[23]),
            'request_time': _float_feature(value=row[24]),
            'lremote_addr': _bytes_feature(value=row[25].encode()),
            'location_city': _bytes_feature(value=row[26].encode(encoding='utf-8'))
        }
        label = row[27]
        feature_dict_list.append((label, feature_dict))
    return feature_dict_list


def _create_rpc_callback(label, result_counter):
    """Creates RPC callback function.

        Args:
        label: The correct label for the predicted example.
        result_counter: Counter for the prediction result.
        Returns:
        The callback function.
    """
    def _callback(result_future):
        """Callback function.

            Calculates the statistics for the prediction result.

            Args:
              result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            response = numpy.array(
                result_future.result().outputs['scores'].float_val)
            prediction = numpy.argmax(response)

            #print(label, prediction)
            if int(label) != int(prediction):
                result_counter.inc_error()
        result_counter.inc_done()
        result_counter.dec_active()
    return _callback


def do_inference(hostport, model, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.

        Args:
        hostport: Host:port address of the PredictionService.
        work_dir: The full path of working directory for test data set.
        concurrency: Maximum number of concurrent requests.
        num_tests: Number of test images to use.

        Returns:
        The classification error rate.

        Raises:
        IOError: An error occurred processing test data set.
    """
    test_feature_dict_list = data_reader(DATA_FILE)
    num_tests = min(num_tests, len(test_feature_dict_list))
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_counter = _ResultCounter(num_tests, concurrency)

    start = time.time()
    for i in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = FLAGS.model
        request.model_spec.signature_name = 'serving_default'

        label, feature_dict = test_feature_dict_list[i]
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()

        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

        result_counter.throttle()
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(
            _create_rpc_callback(label, result_counter))

    print("\nTotal request: %s; Average time: %s" % (num_tests, (time.time() - start)/num_tests))
    return result_counter.get_error_rate()


def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    error_rate = do_inference(FLAGS.server, FLAGS.model,
                            FLAGS.concurrency, FLAGS.num_tests)
    print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
    tf.app.run()
