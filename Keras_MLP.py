import tensorflow as tf


#initialize
neur = tf.keras.models.Sequential()
#layers
neur.add(tf.keras.layers.Dense(units=150, activation='relu'))
neur.add(tf.keras.layers.Dense(units=250, activation='sigmoid'))
neur.add(tf.keras.layers.Dense(units=700, activation='tanh'))

#output layer / no activation for output of regression
neur.add(tf.keras.layers.Dense(units=1, activation=None))

#using mse for regression. Simple and clear
neur.compile(loss='mse', optimizer='adam', metrics=['mse'])

#train
neur.fit(x_subset, y_subset, batch_size=5000, epochs=1000)


#inference
test_out = neur.predict(x_finaleval)
test_out