# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import sys
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from object_detection import ObjectDetection

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'


class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]



def main(image_filename):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFLiteObjectDetection(MODEL_FILENAME, labels)

    image = Image.open(image_filename)
    predictions = od_model.predict_image(image)

    # Draw rectangle
    for pred in predictions:
        if pred['probability'] > .7:
            pred_bound = pred['boundingBox']
            rect_startwith = (pred_bound['left'] * image.width, pred_bound['top'] * image.height)
            pred_shape = [
                rect_startwith, 
                (
                    rect_startwith[0] + pred_bound['width'] * image.width,
                    rect_startwith[1] + pred_bound['height'] * image.height
                )
            ]
            draw_img = ImageDraw.Draw(image)
            draw_img.rectangle(pred_shape, outline='red')

            label = [(pred_shape[0][0], pred_shape[0][1] - 15), (pred_shape[1][0], pred_shape[0][1])]
            draw_img.rectangle(label, fill='red')
            font = ImageFont.truetype("arial.ttf", 16)
            draw_img.text((pred_shape[0][0] + 5, pred_shape[0][1] - 15), pred["tagName"], font=font)

    print(predictions)
    image.show()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
