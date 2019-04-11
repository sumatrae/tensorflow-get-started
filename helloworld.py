import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

#归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu), # 全连接层
  tf.keras.layers.BatchNormalization(),              # 标准化层
  tf.keras.layers.Dropout(0.1),                      # dropout层，drop rate = 0.2
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)# 全连接层
])
model.compile(optimizer='adam',                      # 优化器使用Adam
              loss='sparse_categorical_crossentropy',# 损失函数
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)