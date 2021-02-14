

import argparse
import glob
import os
import numpy as np

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model

from tensorflow.python.lib.io import file_io

import trainer.model as model
from sklearn.model_selection import train_test_split

def train_and_evaluate(args):
    iris_model = model.model_fn()
    # Train the model
    from sklearn.datasets import load_iris
    iris_data = load_iris() # load the iris dataset
    iris_data.data
    _X = iris_data.data
    _y = iris_data.target

    X = _X
    from sklearn.preprocessing import OneHotEncoder 
    ohe = OneHotEncoder()
    y = ohe.fit_transform(np.reshape(_y, (-1, 1))).toarray()

    # which is importance for convergence of the neural network
    #scaler = StandardScaler()
    #X = scaler.fit_transform(_X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=2)
    iris_model.fit(X_train, y_train, verbose=1, batch_size=5, epochs=50)

    # Save model.h5 on to google storage
    iris_model.save('model.h5')
    with file_io.FileIO('model.h5', mode='rb') as input_f:
        with file_io.FileIO(args.job_dir  + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      default='/tmp/census-keras',
      required=True
    )   

    args, _ = parser.parse_known_args()
    train_and_evaluate(args)

