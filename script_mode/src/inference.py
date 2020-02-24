import io
import json
import urllib
import numpy as np
from PIL import Image
import logging

LABELS = ['air_conditioner',
 'air_cooler',
 'camera',
 'desktop',
 'geyser',
 'ipod_mp3_player',
 'laptop',
 'microwave_oven',
 'music_system',
 'refrigerator',
 'tv',
 'video_game',
 'washing_machine',
 'water_purifier']

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == 'application/json':
        d = json.loads(data.read().decode('utf-8'))
        imageURLs = d['image_urls']
        imageArrayList = []
        for imageURL in imageURLs:
            image = _loadImageV2(imageURL)
            imageArrayList.append(image)
        np.array(imageArrayList).shape
        
        json_req = json.dumps({"signature_name": "serving_default",
                          "instances": np.array(imageArrayList).tolist()})
        logging.info(json_req)
        return json_req

    _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise Exception(data.content.decode('utf-8'))
    response_content_type = context.accept_header
    prediction = data.json()
    logging.info(data.content)
    logging.info(data.content.decode('utf-8'))
    logging.info(prediction)

    return _processOutput(prediction['predictions']), response_content_type



def _processOutput(predictions):
    logging.info(type(predictions))
    logging.info(predictions)
    instance_prediction = []
    for prediction in predictions:
        logging.info(prediction)
        logging.info(type(prediction))
        top_labels = []
        sorted_label = np.flip(np.argsort(prediction))[:3]
        for index in sorted_label:
            current_label = {
                "label": LABELS[index],
                "score": prediction[index]
            }
            top_labels.append(current_label)
        instance_prediction.append({"top_classes": top_labels})

    return json.dumps({"predictions": instance_prediction})

# def _loadImage(URL):
#     with urllib.request.urlopen(URL) as url:
#         img = keras.preprocessing.image.load_img(io.BytesIO(url.read()), target_size=(160, 160))

#     return keras.preprocessing.image.img_to_array(img) * 1./255

def _loadImageV2(url):
    pil_image = Image.open(urllib.request.urlopen(url)).convert("RGB")
    pil_image = pil_image.resize((160,160))
    pil_array = np.array(pil_image)
    return pil_array * 1./255


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))