
import struct
import time
import pickle

# office Shelves

# rack_shelves_sensors = {
#     "rack_1": {
#         "cameras": [7, 3],
#         "shelves":{
#             "shelf_1": {
#                 "sensors": [0],
#                 "products": {"britesurf": 1015, "closeup": 188},
#                 "location": {7: (437, 146), 3: (312, 74)}
#                 },
#             "shelf_2": {
#                 "sensors": [1],
#                 "products": {"olpers": 1060, "spritecan": 275, "fantacan": 275, "pepsican": 275},
#                 "location": {7: (443, 232), 3: (353, 177)}
#                 },
#             "shelf_3": {
#                 "sensors": [2],
#                 "products": {"lifebouyshampo": 420, "pepsican": 275},
#                 "location": {7: (446, 305), 3: (377, 246)}
#                 },
#             # "shelf_4": {
#             #     "sensors": [3],
#             #     "products": {"lifebouyshampo": 420, "pepsican": 275},
#             #     "location": {7: (438, 191), 3: (682, 158)}
#             #     },
#             },  
#         },
#     # "rack_2": {
#     #     "cameras": [3, 4],
#     #     "shelves":{
#     #         "shelf_1": {
#     #             "sensors": [2, 3],
#     #             "products": {"can": 275, "bottle": 540},
#     #             "location": {3: (200,300), 7: (200,300)}
#     #             },
#     #         },  
#     #     },
# }


# Haris sahb shelves

# rack_shelves_sensors = {
#     "rack_1": {
#         "cameras": [401, 501],
#         "shelves":{
#             "shelf_1": {
#                 "sensors": [0],
#                 "products": {"First Price Bearnaise Sauce": 129, "first price flydende":500},
#                 "location": {401: (247, 103), 501: (501, 118)}
#                 },
#             "shelf_2": {
#                 "sensors": [1],
#                 "products": {"first price kartoffel mos pulver": 391, "Salina Salt": 1013},
#                 "location": {401: (263, 178), 501: (510, 217)}
#                 },
#             "shelf_3": {
#                 "sensors": [2],
#                 "products": {"First price tomat ketchup": 1066, "first price remoulade": 433 , "lasagne gialle":512},
#                 "location": {401: (300, 235), 501: (511, 287)}
#                 },
#             # "shelf_4": {
#             #     "sensors": [3],
#             #     "products": {"lifebouyshampo": 420, "pepsican": 275},
#             #     "location": {7: (438, 191), 3: (682, 158)}
#             #     },
#             },  
#         },
#     # "rack_2": {
#     #     "cameras": [3, 4],
#     #     "shelves":{
#     #         "shelf_1": {
#     #             "sensors": [2, 3],
#     #             "products": {"can": 275, "bottle": 540},
#     #             "location": {3: (200,300), 7: (200,300)}
#     #             },
#     #         },  
#     #     },
# }

# Vester haisenge shelves

rack_shelves_sensors = {
    "rack_1": {
        "cameras": [21, 9],
        "shelves":{
            "shelf_1": {
                "sensors": [0],
                "products": {"Natron (bicarbonate of soda)": 163, "First price spaghetti": 1050, "Vat Rondeller": 41},
                "location": {21: (247, 103), 9: (501, 118)}
                },
            "shelf_2": {
                "sensors": [1],
                "products": {"Salina Salt": 1068, "first price flydende": 496, "First Price Peanuts (ristede og saltede)": 257},
                "location": {21: (263, 178), 9: (510, 217)}
                },
            "shelf_3": {
                "sensors": [2],
                "products": {"Snacks Med bacon smag": 159, "First Price Fusilli torret Pasta": 493, "First Price Aeblemos": 1028},
                "location": {21: (300, 235), 9: (511, 287)}
                },
            "shelf_4": {
                "sensors": [3],
                "products": {"Pepsi Max": 526, "Banderos Tortilla Chips Salted": 203, "Gron balance Body Lotion": 377},
                "location": {21: (438, 191), 9: (682, 158)}
                },
            },  
        },
    "rack_2": {
        "cameras": [21, 9],
        "shelves":{
            "shelf_1": {
                "sensors": [4],
                "products": {"Oksebouillon": 131, "first price kartoffel mos pulver": 391, "first price remoulade": 434}, # note down ("Nik Naks (the double crunh peanuts)")
                "location": {21: (200,300), 9: (200,300)}
                },
            "shelf_2": {
                "sensors": [5],
                "products": {"Fuldkorns spaghetti (af durumhvede)":501, "First Price Bearnaise Sauce": 113, "maizena majsstiveslse majsstarkelse": 417},
                "location": {21: (200,300), 9: (200,300)}
                },
            "shelf_3": {
                "sensors": [6],
                "products": {"First price lommeletter": 217, "First price tomat ketchup": 1058, "lasagne gialle": 512},
                "location": {21: (200,300), 9: (200,300)}
                },
            "shelf_4": {
                "sensors": [7],
                "products": {"First Price Afspaendings middle": 535, "AJAX Universal (3x easy Cleaning)": 818, "First price rengorings svampe (med skureside)": 44},
                "location": {21: (200,300), 9: (200,300)}
                },
            "shelf_5": {
                "sensors": [8],
                "products": {"Bind normal (pads)": 206, "First Price Fryseposer 4 liter (100 STK.)": 384, "First price sorte oliven (uden sten)": 512},
                "location": {21: (200,300), 9: (200,300)}
                },
            },  
        },
}



def map_sensors_to_values(rack_data, sensor_values):
    index = 0
    for rack in rack_data.values():
        for shelf in rack.get("shelves", {}).values():
            sensors = shelf.get("sensors", [])
            # Update sensor values from sensor_values list
            for i in range(len(sensors)):
                if index < len(sensor_values):
                    sensors[i] = sensor_values[index]
                else:
                    sensors[i] = 0  # Fill remaining sensors with zero
                index += 1

    return rack_data



def create_shelves_stats(rack_shelves_sensors=rack_shelves_sensors):
    shelves_state = {
    f"{rack_id}_{shelf_id}": {
        "prev_weight": None,
        "action_start_time": None,
        "state": "stable",
        "weight_history": [],
        "initial_weight": None,
    }
    for rack_id, rack in rack_shelves_sensors.items()
    for shelf_id in rack.get("shelves", {}).keys()
    }
    return shelves_state


# def hexArrayToByteStream(hex_array):
#     return bytes(hex_array)

# def decode_weight(weight_bytes):
#     # Interpret bytes as a signed 32-bit integer
#     weight = struct.unpack('<i', bytes(weight_bytes))[0]
#     weight_in_grams = weight * 0.1
#     return weight_in_grams



def map_sensors_to_values(rack_data, sensor_values):
    index = 0
    for rack in rack_data.values():
        for shelf in rack.get("shelves", {}).values():
            sensors = shelf.get("sensors", [])
            # Update sensor values from sensor_values list
            for i in range(len(sensors)):
                if index < len(sensor_values):
                    sensors[i] = sensor_values[index]
                else:
                    sensors[i] = 0  # Fill remaining sensors with zero
                index += 1

    return rack_data



def create_shelves_stats(rack_shelves_sensors=rack_shelves_sensors):
    shelves_state = {
    f"{rack_id}_{shelf_id}": {
        "prev_weight": None,
        "action_start_time": None,
        "state": "stable",
        "weight_history": [],
        "initial_weight": None,
    }
    for rack_id, rack in rack_shelves_sensors.items()
    for shelf_id in rack.get("shelves", {}).keys()
    }
    return shelves_state


def identify_product(weight_difference, products):
    possible_pickup_product = []
    for product, product_weight in products.items():
        # Calculate the allowed deviation (5% of the product weight)
        allowed_deviation = product_weight * 0.07
        # Check if the weight difference matches a product
        if abs(weight_difference - product_weight) <= allowed_deviation:
            possible_pickup_product.append(product)
    if len(possible_pickup_product) !=0 :
        return possible_pickup_product
    else:
        return None


def decode_weight(weight_bytes):
    weight = weight_bytes[0] + (weight_bytes[1] << 8) + (weight_bytes[2] << 16) + (weight_bytes[3] << 24)
    return weight * 0.1


def tare_sensor(client_socket):
    response = bytes([0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03])
    client_socket.sendall(response)
    
    msg = client_socket.recv(1024)
    if msg:
        return True
    else: 
        return False


def generate_hex_command(aisle_id, sensors_per_lane):
    if not (1 <= aisle_id <= 64):
        raise ValueError("Aisle ID must be between 1 and 64")
    
    if sensors_per_lane not in [1, 2, 4]:
        raise ValueError("Sensors per lane must be 1, 2, or 4")

    # Fixed bytes
    command = [0x02, 0x01, 0x05, 0x00, 0x57, 0x50, 0x05]
    # Dynamic bytes
    command.append(aisle_id)  # Aisle ID
    command.append(sensors_per_lane)  # Sensors per lane
    # Checksum calculation (sum of all bytes after start byte, modulo 256)
    checksum = sum(command[1:]) % 256
    command.append(checksum)  # Adding checksum
    # End byte
    command.append(0x03)

    return command



def setup_aisle_configuration(client_socket, aisleId, sensorsPerLane):
    command = generate_hex_command(aisleId, sensorsPerLane)
    response = bytes(command)
    client_socket.sendall(response)
    
    msg = client_socket.recv(1024)
    if msg:
        return True
    else: 
        return False


def process_single_client(client_socket, numAisles, rack_shelves_sensors=rack_shelves_sensors):
    try:
        response = bytes([0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03])
        client_socket.sendall(response)
        msg = client_socket.recv(1024)

        if msg:
            hex_a = list(msg)
            # print(hex_a)
            data_section = hex_a[8:-2] # the value can be 9 or 8
            # data_section = hex_a[8:-2]
            # print(data_section)

            sensor_weights = []
            for i in range(numAisles):
                start = i * 4
                end = start + 4
                acc_weight = decode_weight(data_section[start:end])
                if len(str(acc_weight)) <= 7:
                    sensor_weights.append(acc_weight)
                else:
                    sensor_weights.append(0.0)

                # sensor_weights.append(decode_weight(data_section[start:end]))
            
            total_weight = map_sensors_to_values(rack_shelves_sensors, sensor_weights)

            return total_weight
            
    except Exception as e:
        print("Error: ", e)
        return []
        # client_socket.close()
            



