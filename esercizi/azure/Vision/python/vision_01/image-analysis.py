from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
from azure.core.exceptions import HttpResponseError
import requests

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential



# Import namespaces


def main():
    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # Get image
        image_file = './images/street.jpg'
     

        with open(image_file, "rb") as f:
            image_data = f.read()

        # Authenticate Azure AI Vision client
        cv_client = ImageAnalysisClient(
            endpoint   = ai_endpoint,
            credential = AzureKeyCredential(ai_key)
        )
        
        # Analyze image
        AnalyzeImage(image_file, image_data, cv_client)
        
    except Exception as ex:
        print(ex)


def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Get result with specified features to be retrieved
        result = cv_client.analyze(
            image_data = image_data, 
            visual_features = [
                #VisualFeatures.CAPTION,
                #VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE
            ]
        )
        print(result)

        if result.caption is not None:
            print("\nCaption:")
            print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

        if result.dense_captions is not None:
            print("\nDense Captions:")
            for caption in result.dense_captions.list:
                print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))        

    except HttpResponseError as e:
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

    # Display analysis results
    


if __name__ == "__main__":
    main()
