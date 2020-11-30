#
# Credit: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb
# 


import tensorflow as tf

#
# HYPER PARAMS
#
number_classes = 10
input_width = 28
input_height = 28
num_hidden_layer = 5


#
# Load data (MNIST)
#
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x \in [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0


#
# CREATE MODEL
#
network_layers = []
network_layers.append(tf.keras.layers.Flatten(input_shape=(input_height, input_width)))

for _ in range(num_hidden_layer):
    network_layers.append(tf.keras.layers.Dense(128, activation='relu'))

network_layers.append(tf.keras.layers.Dense(number_classes))

model = tf.keras.models.Sequential(network_layers)

# Show some predictions of the untrained model
predictions = model(x_train[:1]).numpy()
softmax_predictions = tf.nn.softmax(predictions).numpy()

print(predictions)
print(softmax_predictions)


#
# Define the loss function i.e. our "goal"
#
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#
# TRAINING LOOP - Total of 25 epochs
#
for _ in range(5):
    model.fit(x_train, y_train, epochs=5)

    # And teset it on the test set
    model.evaluate(x_test,  y_test, verbose=2)


# Lets print some predictions of the trained model
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:1]))