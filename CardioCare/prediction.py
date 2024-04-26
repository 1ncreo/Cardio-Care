import tensorflow as tf
model = tf.keras.models.load_model('/root/AI proj/saved_model.h5')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/root/AI proj/data/predict/LBBB.png',target_size=(64,64))

test_image = image.img_to_array(test_image)
test_image /= 255.0
test_image = np.expand_dims(test_image, axis = 0)
predictions = model.predict(test_image)
predicted_class_index = np.argmax(predictions)

class_labels = [
    'left Bundle Branch block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contraction',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]
predicted_class_label=class_labels[predicted_class_index]
print(predicted_class_label)