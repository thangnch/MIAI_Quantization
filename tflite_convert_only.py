
import tensorflow as tf

# Load model keras
keras_model = tf.keras.models.load_model("saved_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quantize = converter.convert()


with open("tflite_model_quantize.tflite", "wb") as f:
    f.write(tflite_model_quantize)


