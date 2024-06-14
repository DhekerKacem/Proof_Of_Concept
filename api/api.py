from flask import Flask, request, send_file
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from transformers import SegformerForSemanticSegmentation
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import backend as K
import keras
from PIL import Image

app = Flask(__name__)

# Définir la palette de couleurs pour les classes
PALETTE = np.array([
    [0, 0, 0],        # void : Noir
    [128, 64, 128],   # flat : Violet
    [70, 70, 70],     # construction : Gris foncé
    [190, 153, 153],  # object : Rose clair
    [107, 142, 35],   # nature : Vert
    [70, 130, 180],   # sky : Bleu ciel
    [220, 20, 60],    # human : Rouge
    [0, 0, 142]       # vehicle : Bleu foncé
])

def preprocess_image_seg(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1024, 1024))  # Prétraiter l'image à la taille attendue par le modèle
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

def convert_seg_mask_to_color(mask):
    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for i in range(8):
        color_mask[mask[0] == i] = PALETTE[i]
    return color_mask

def overlay_mask_on_image(image_bytes, mask):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    overlay_image = cv2.addWeighted(original_img, 1, mask, 0.5, 0)
    return overlay_image

@app.route("/predict-mask-seg", methods=["POST"])
def predict_mask_seg():    
    model_seg = SegformerForSemanticSegmentation.from_pretrained('../models/seg')
    model_seg.eval()

    image_bytes = request.data
    original_image = Image.open(BytesIO(image_bytes))
    original_dims = original_image.size

    image_tensor = preprocess_image_seg(image_bytes).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model_seg(image_tensor)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_mask = torch.argmax(probs, dim=1).cpu()

    predicted_mask_resized = F.interpolate(predicted_mask.float().unsqueeze(0), size=original_dims[::-1], mode='nearest').squeeze(0)
    predicted_mask_color = convert_seg_mask_to_color(predicted_mask_resized.numpy())

    overlay_image = overlay_mask_on_image(image_bytes, predicted_mask_color)
    output = Image.fromarray(overlay_image)
    output_bytes = BytesIO()
    output.save(output_bytes, format='PNG')
    output_bytes.seek(0)

    return send_file(output_bytes, mimetype='image/png', as_attachment=True, download_name='predicted_mask.png')

# Définir et enregistrer la métrique IoU personnalisée pour TensorFlow/Keras
@keras.saving.register_keras_serializable(package="MyMetrics")
class CustomMeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name='mean_iou', **kwargs):
        super(CustomMeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count_classes = self.add_weight(name='count_classes', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 4:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)

        mean_iou = 0.0
        class_count = 0.0

        for i in range(self.num_classes):
            true_class = tf.cast(tf.equal(y_true, i), dtype=tf.float32)
            pred_class = tf.cast(tf.equal(y_pred, i), dtype=tf.float32)

            intersection = tf.reduce_sum(true_class * pred_class)
            union = tf.reduce_sum(true_class) + tf.reduce_sum(pred_class) - intersection

            iou = intersection / (union + K.epsilon())
            condition = tf.equal(union, 0)
            mean_iou = tf.where(condition, mean_iou, mean_iou + iou)
            class_count = tf.where(condition, class_count, class_count + 1)

        self.total_iou.assign_add(mean_iou)
        self.count_classes.assign_add(class_count)

    def result(self):
        return self.total_iou / (self.count_classes + K.epsilon())

    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count_classes.assign(0.0)

    def get_config(self):
        config = super(CustomMeanIoU, self).get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@app.route("/predict-mask-unet", methods=["POST"])
def predict_mask_unet():
    image_bytes = process_mask_unet(request.data)
    return send_file(BytesIO(image_bytes), mimetype='image/png', as_attachment=True, download_name='predicted_mask.png')

def preprocess_image_unet(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    size_x, size_y = img.shape[1], img.shape[0]
    img = cv2.resize(img, (512, 512))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  
    return img, size_x, size_y

def convert_unet_mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(8):
        color_mask[mask == i] = PALETTE[i]
    return color_mask

def process_mask_unet(raw_data):
    image, size_x, size_y = preprocess_image_unet(raw_data)
    model_unet = keras.models.load_model("../models/unet/mini_unet_hd_complete.h5", custom_objects={"CustomMeanIoU": CustomMeanIoU})
    prediction = model_unet.predict(image)
    predicted_mask = np.argmax(prediction, axis=-1).squeeze()  # Supprimer la dimension de batch
    
    print("Predicted mask values:", np.unique(predicted_mask))  # Ajout de l'impression des valeurs uniques du masque

    # Convertir le masque prédit en une image colorée
    predicted_mask_color = convert_unet_mask_to_color(predicted_mask)

    print("Unique colors in the color mask:", np.unique(predicted_mask_color.reshape(-1, predicted_mask_color.shape[2]), axis=0))  # Vérifier les couleurs uniques

    # Redimensionner le masque coloré à la taille originale de l'image
    predicted_mask_color = cv2.resize(predicted_mask_color, (size_x, size_y), interpolation=cv2.INTER_NEAREST)

    # Charger l'image originale
    nparr = np.frombuffer(raw_data, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Superposer le masque coloré sur l'image originale
    mask_image = Image.fromarray(predicted_mask_color)
    mask_image.putalpha(120)  # Ajouter la transparence
    original_img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    original_img_pil.paste(mask_image, (0, 0), mask_image)

    # Sauvegarder l'image finale avec le masque superposé
    output = BytesIO()
    original_img_pil.save(output, format='PNG')
    output.seek(0)

    return output.getvalue()

if __name__ == "__main__":
    app.run(debug=True)
