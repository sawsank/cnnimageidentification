import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import mobilenet_v2, resnet50
import PIL.Image as Image

class ImageClassifier:
    def __init__(self, model_name='mobilenet_v2'):
        self.model_name = model_name
        if model_name == 'mobilenet_v2':
            self.model = mobilenet_v2.MobileNetV2(weights='imagenet')
            self.preprocess_input = mobilenet_v2.preprocess_input
            self.decode_predictions = mobilenet_v2.decode_predictions
            self.target_size = (224, 224)
        elif model_name == 'resnet50':
            self.model = resnet50.ResNet50(weights='imagenet')
            self.preprocess_input = resnet50.preprocess_input
            self.decode_predictions = resnet50.decode_predictions
            self.target_size = (224, 224)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def predict(self, img_path):
        img = Image.open(img_path).resize(self.target_size)
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)

        preds = self.model.predict(x)
        return self.decode_predictions(preds, top=5)[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="mobilenet_v2")
    args = parser.parse_args()

    classifier = ImageClassifier(args.model)
    results = classifier.predict(args.image)
    print(f"Predictions for {args.image}:")
    for _, label, score in results:
        print(f"{label}: {score:.4f}")
