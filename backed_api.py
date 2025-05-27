# import cv2
# import numpy as np
import json
# import os
# import time
# import torch
# from ultralytics import YOLO
# import cvzone
# import random
# import asyncio
# import websockets
import requests
# import threading


###### Dev Product Urls ##########

# vendor_id = '34126bc9-ad96-4bbc-a2cd-42c18e62db97'
# product_data_url = 'https://dev.api.admin.unlimitretail.dk/api/app-settings/fetch-vendor-products/'
# login_url = 'https://dev.api.admin.unlimitretail.dk/api/third-party/login'
# login_data = {
#     "email": "aiuser@sa.com",
#     "password": "12345678",
#     "abilities": ["fetch-vendor-products"]
# }



###### Staging ##########

vendor_id = 'bc7404f9-be39-429d-9f54-51f16c952c09'
product_data_url = 'https://adminapi.stg.unlimitretail.dk/api/app-settings/fetch-vendor-products/'
login_url = 'https://adminapi.stg.unlimitretail.dk/api/third-party/login'
login_data = {
    "email":"aiuser@sa.com",
    "password":"12345678",
    "abilities":["fetch-vendor-products"]
}



# shop_id = "3c039929-be4e-49a8-b38d-5ab6a55dee51" # Dev

# shop_id = "edf8be0b-0441-491c-9087-29160b14a1ee" # Staging Near shop
# shop_id = "2135fa3c-4299-44b4-be57-5721d01dd7f3" # New staging Shop id (when WFH)

# shop_id = "5512f883-065a-4279-97ef-0a41525c1ff0" # haris sahb office shop

shop_id = "2135fa3c-4299-44b4-be57-5721d01dd7f3" # Vester haisenge Shop


# product_list = {
#     "bottle":
#         {
#         # "id": "dac0c107-7580-446c-b690-82d892c9b08f", # Dev
#         "id": "8081df90-9ad3-4ddf-903a-344a97ebefd8", # Staging
#         "name": "Fanta Orange PET 50 Cl.*",
#         "image_url": "https://stg-unlimitretail.s3.eu-north-1.amazonaws.com/products/attachment/PetFantaOrange_50Cl_Pet_1724072811.PNG",
#         "unit": "ml",
#         "size": "500",
#         "sale_price": "27.25",
#         "original_price": "28.50",
#         "currency": "kr",
#         "points": 0,
#         "quantity": 1
#         },
#     "can":
#         {
#         # "id": "0b3f5783-f2ed-471e-ad04-88505b2e79a3", # Dev
#         "id": "1865b09c-9fb7-4763-ba53-97619a8c9c08", # Staging
#         "name": "Pepsi",
#         "image_url": "https://stg-unlimitretail.s3.eu-north-1.amazonaws.com/products/attachment/1029886380_1724744228.jpg",
#         "unit": "ml",
#         "size": "60",
#         "sale_price": "24.45",
#         "original_price": "30.50",
#         "currency": "kr",
#         "points": 0,
#         "quantity": 1
#         },
#     "wavy":
#         {
#         # "id": "0b3f5783-f2ed-471e-ad04-88505b2e79a3", # Dev
#         "id": "3b0c695e-1950-4c0a-90fb-124baaf7389e", # Staging
#         "name": "Lay's Wavy BBQ Potato Chips",
#         "image_url": "https://stg-unlimitretail.s3.eu-north-1.amazonaws.com/products/attachment/1_1724744652.jpg",
#         "unit": "GR",
#         "size": "50",
#         "sale_price": "32",
#         "original_price": "40",
#         "currency": "kr",
#         "points": 0,
#         "quantity": 1
#         },
# }




def get_userId(shop_id=shop_id):
    # url = 'https://dev.api.admin.unlimitretail.dk/api/getCustomerId' # Dev
    url = 'https://adminapi.stg.unlimitretail.dk/api/getCustomerId' # staging
    payload = {
        'shop_id': shop_id
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        customer_id = data.get('customer_id')
        return customer_id
        # print(f"User ID: {user_id}")


def check_out(userId, shop_id=shop_id):
    # url = 'https://socket.dev.unlimitretail.dk:3000/api/cart/checkout' # Dev
    url = 'https://socket.unlimitretail.dk:3000/api/cart/checkout' # Staging
    payload = {
        'user_id':userId,
        'shop_id':shop_id
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        flag = data.get('success')
        return flag


def fetch_product_data(r_products, login_data=login_data, login_url=login_url, product_data_url=product_data_url, vendor_id=vendor_id):
    try:
        # Step 1: Log in and get the Bearer token
        login_response = requests.post(login_url, json=login_data)
        # login_response.raise_for_status()
            
        login_data = login_response.json()
        bearer_token = login_data['data']['token']
        # bearer_token = '1914|frHmgW354yNeHVXK8h7TeCi6umTRBPG18A1wTYFu6321cfa4' # staging
        # bearer_token = '656|8B3ZU9taFbDD24l9HQFZfdZfQOOvWuR9M72Eovqg247b36e4' # Dev
        # Step 2: Use the Bearer token to fetch product data
        headers = {'Authorization': f'Bearer {bearer_token}'}
        product_response = requests.get(f"{product_data_url}{vendor_id}", headers=headers)
        product_response.raise_for_status()
        
        product_data = product_response.json()
        product_data_dictionary = product_data['data'][0]['content']

        r_products.set("product_data", json.dumps(product_data_dictionary))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

        # time.sleep(20)


def load_product_data(r_products, key):
    try:
        json_data = r_products.get('product_data')
        product_data = json.loads(json_data)
        return product_data[key]
    except Exception as e:
        print(e)
        return {}