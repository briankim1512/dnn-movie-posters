## About
Use Convolutional Neural Network (CNN) to classify movies posters by genre. It is a multi-label classification problem 
(movies can belong to multiple genres). Each instance (movie poster) has an independent probability to belong to each label (genre).

The implementation is based on Keras and TensorFlow.

[Original Repo](https://github.com/benckx/dnn-movie-posters) by benckx

## Installation

### Requirements
* Linux
    * imagemagick
* Python
    * Python 3.5
    * [tensorflow 1.5.0](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv)
    * [Keras 2.1.4](https://keras.io/#installation)
    * pandas 0.22.0
    * h5py 2.7.1

To install the linux requirement (imagemagick) run this command in the terminal
```apt install imagemagick```

To install the python requirements, simply run the following command whilst in the repo folder

```pip install -r requirements.txt```

### Get posters data
* Use flag `-download` to download the posters from Amazon (based on the URLs provided in MovieGenre.csv)
* Use flag `-resize` to create smaller posters (30%, 40%, etc)
* Use parameter `-min_year=1980` to filter out the oldest movies
```
python3 get_data.py -download -resize
```

## Training & Testing

### Train the model
This script builds and trains models. Models are saved to 'saved_models'. One or multiple models
(with different parameters) can be produced.
```
python3 __main__.py
```

### Evaluate the model and test predictions
This script iterates through all the saved models in 'saved_models' and evaluates them on the test data.
```
python3 tests.py
```

## Config

The configuration for the network can be found on line `60` to line `77` from the file `movies_genre_model.py`
The config is represented below.

```python
model = Sequential([
    Conv2D(32, kernel_dimensions1, padding='same', input_shape=x_train.shape[1:], activation='relu'),
    Conv2D(32, kernel_dimensions1, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_dimensions2, padding='same', activation='relu'),
    Conv2D(64, kernel_dimensions2, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')
])

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
```

If you wish to change the number of epochs the training goes through, you can do so by changing it in `__main__.py` on line `6`