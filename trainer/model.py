from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def model_fn():
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())
    return model



