# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import time

import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'DNN',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
    csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
    for row in csv_reader:
        yield [unicode(cell, 'utf-8') for cell in row]

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'


    with open('/Users/zsc/ml/DNN/tf_dataset_and_estimator_apis/dataset/security_test.csv', 'rb') as f:
        reader = unicode_csv_reader(f)
        data_list = list(reader)

    accu = 0
    neg = 0
    start = time.time()
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

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        serialized = example.SerializeToString()

        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

        result_future = stub.Predict.future(request, 5.0)
        prediction = result_future.result().outputs['scores']

        # print('True label: ' + str(label))
        # print('Prediction: ' + str(np.argmax(prediction.float_val)))
        if int(label) == int(np.argmax(prediction.float_val)):
            accu += 1
        else:
            neg += 1

    print("Precision: ", accu / (accu + neg))
    print("Speed: ", (time.time() - start) / (accu + neg))

if __name__ == '__main__':
    tf.app.run()