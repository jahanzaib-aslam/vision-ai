
########### Zohaib script ................

############ wokring for 1
# import time
# import socket
# import threading
# import datetime
# import struct

# # Specify server host and port
# HOST = "0.0.0.0"
# PORT = 8235

# # Products and their weights
# products = {"CAN": 275, "BOTTLE": 536, "CHIPS": 43}


# def hexArrayToByteStream(hex_array):
#     return bytes(hex_array)


# def decode_weight(weight_bytes):
#     # Make sure we have exactly 4 bytes for unpacking
#     if len(weight_bytes) != 4:
#         # Pad the array if needed or return 0 for invalid data
#         if len(weight_bytes) < 4:
#             weight_bytes = weight_bytes + [0] * (4 - len(weight_bytes))
#         else:
#             weight_bytes = weight_bytes[:4]

#     # Interpret bytes as a signed 32-bit integer
#     weight = struct.unpack("<i", bytes(weight_bytes))[0]
#     weight_in_grams = weight * 0.1
#     return weight_in_grams


# def identify_product(weight_difference):
#     for product, product_weight in products.items():
#         # Calculate the allowed deviation (5% of the product weight)
#         allowed_deviation = product_weight * 0.05
#         # Check if the weight difference matches a product
#         if abs(weight_difference - product_weight) <= allowed_deviation:
#             return product
#     return None  # Return None for unknown products


# def send_command_and_handle_response(
#     client_socket, command, num_aisles, units_per_aisle
# ):
#     global getweight_counter
#     response = hexArrayToByteStream(command)
#     client_socket.sendall(response)

#     msg = client_socket.recv(1024)
#     # print(f"{msg=}")
#     if msg:
#         hex_a = list(msg)
#         # print(f"{hex_a=}")

#         # Different response handling based on aisle and unit configuration
#         weights = []

#         try:
#             # For configurations with multiple aisles/units, the response format changes
#             # Extract weights depending on the number of aisles and units per aisle
#             if len(hex_a) >= 14:  # Make sure there's enough data
#                 # For single aisle, single unit
#                 if num_aisles == 1 and units_per_aisle == 1:
#                     if len(hex_a) >= 12:
#                         weights.append(decode_weight(hex_a[8:12]))

#                 # For single aisle, multiple units
#                 elif num_aisles == 1 and units_per_aisle > 1:
#                     for i in range(units_per_aisle):
#                         start_idx = 8 + (i * 4)
#                         if (
#                             start_idx + 4 <= len(hex_a) - 2
#                         ):  # Check if we have enough data
#                             weights.append(
#                                 decode_weight(hex_a[start_idx : start_idx + 4])
#                             )

#                 # For multiple aisles
#                 else:
#                     for i in range(num_aisles):
#                         for j in range(units_per_aisle):
#                             start_idx = 8 + (i * units_per_aisle * 4) + (j * 4)
#                             if (
#                                 start_idx + 4 <= len(hex_a) - 2
#                             ):  # Check if we have enough data
#                                 weights.append(
#                                     decode_weight(hex_a[start_idx : start_idx + 4])
#                                 )

#             # Sum up all the weights
#             total_weight = sum(weights) if weights else 0

#             if command[5] == 0x50 and command[6] == 0x06:  # TARE command
#                 print(f"Tare completed. Found {len(weights)} weight sensors.")

#             if command[5] == 0x50 and command[6] == 0x06:  # GET_WEIGHT command
#                 getweight_counter += 1

#             return total_weight, weights

#         except Exception as e:
#             print(f"Error processing weights: {e}")
#             return 0, []

#     return 0, []


# def handle_client(client_socket, num_aisles, units_per_aisle):
#     global getweight_counter
#     prev_weight = None
#     action_start_time = None
#     state = "stable"
#     MIN_WEIGHT_CHANGE = 5  # grams
#     STABILIZATION_TIME = 1.0  # seconds
#     MAX_WAIT_TIME = 3.0  # seconds
#     WEIGHT_HISTORY_SIZE = 25
#     STABILITY_THRESHOLD = 2  # grams

#     # Perform initial TARE operation
#     total_weight, individual_weights = send_command_and_handle_response(
#         client_socket, TARE, num_aisles, units_per_aisle
#     )
#     print(f"Initial weights: {individual_weights}")

#     weight_history = []
#     initial_weight = None

#     while True:
#         try:
#             # Get the current weight from the scales
#             current_weight, individual_weights = send_command_and_handle_response(
#                 client_socket, GET_WEIGHT, num_aisles, units_per_aisle
#             )
#             print(f"Current individual weights: {individual_weights}")

#             weight_history.append(current_weight)
#             if len(weight_history) > WEIGHT_HISTORY_SIZE:
#                 weight_history.pop(0)

#             if prev_weight is None:
#                 prev_weight = current_weight
#                 initial_weight = current_weight
#                 continue

#             weight_difference = current_weight - prev_weight

#             if abs(weight_difference) >= MIN_WEIGHT_CHANGE:
#                 if state == "stable":
#                     action_start_time = time.time()
#                     state = "action"
#                     initial_weight = prev_weight

#             current_time = time.time()
#             if state == "action":
#                 if current_time - action_start_time >= STABILIZATION_TIME:
#                     if max(weight_history) - min(weight_history) < STABILITY_THRESHOLD:
#                         total_change = current_weight - initial_weight
#                         action_type = "picked" if total_change < 0 else "dropped"
#                         product = identify_product(abs(total_change))
#                         if product:
#                             print(
#                                 f"Product {action_type}: {product}, Weight change: {abs(total_change):.1f}g, Current weight: {current_weight:.1f}g; #:{round(total_change/products[product])}"
#                             )
#                             getweight_counter = 0

#                         state = "stable"
#                         weight_history.clear()
#                 elif current_time - action_start_time >= MAX_WAIT_TIME:
#                     state = "stable"
#                     weight_history.clear()

#             prev_weight = current_weight
#             time.sleep(0.2)  # Add a small delay to prevent tight loops

#         except Exception as e:
#             print(f"Error in handle_client: {e}")
#             if client_socket in client_sockets:
#                 client_sockets.remove(client_socket)
#             client_socket.close()
#             break


# def start_server(num_aisles, units_per_aisle):
#     global TARE, GET_WEIGHT, CONFIG_SCALE, getweight_counter, client_sockets

#     # Reset global variables
#     getweight_counter = -1
#     client_sockets = []

#     # Commands
#     GET_WEIGHT = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xAC, 0x03]
#     TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]

#     # Create CONFIG_SCALE with the specified parameters
#     # Format: 02 01 05 00 57 50 05 [num_aisles] [units_per_aisle] [checksum] 03
#     CONFIG_SCALE = [
#         0x02,
#         0x01,
#         0x05,
#         0x00,
#         0x57,
#         0x50,
#         0x05,
#         num_aisles,
#         units_per_aisle,
#     ]

#     # Calculate checksum (simple sum of bytes 1 to n-1)
#     checksum = sum(CONFIG_SCALE[1:]) % 256
#     CONFIG_SCALE.append(checksum)
#     CONFIG_SCALE.append(0x03)

#     print("server ip:", HOST)
#     print("server port:", PORT)
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.setsockopt(
#         socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
#     )  # Allow reuse of address

#     try:
#         server_socket.bind((HOST, PORT))
#         server_socket.listen(5)
#         print("Waiting for client connection...")

#         while True:
#             client_socket, addr = server_socket.accept()
#             print(f"Connecting to client {addr}")
#             client_sockets.append(client_socket)

#             # Send configuration command
#             print(f"Using configuration: {[hex(x) for x in CONFIG_SCALE]}")
#             client_socket.sendall(hexArrayToByteStream(CONFIG_SCALE))
#             response = client_socket.recv(1024)
#             print(f"Configuration response: {list(response)}")

#             # Start client handling thread
#             thread = threading.Thread(
#                 target=handle_client, args=(client_socket, num_aisles, units_per_aisle)
#             )
#             thread.daemon = True
#             thread.start()

#     except KeyboardInterrupt:
#         print("\nShutting down server...")
#     except Exception as e:
#         print(f"Server error: {e}")
#     finally:
#         for client in client_sockets:
#             try:
#                 client.close()
#             except:
#                 pass
#         server_socket.close()


# if __name__ == "__main__":
#     print("Scale Configuration Utility")
#     print("---------------------------")

#     while True:
#         try:
#             num_aisles = int(input("Enter number of aisles (1-64): "))
#             units_per_aisle = int(input("Enter units per aisle (1-4): "))

#             if 1 <= num_aisles <= 64 and 1 <= units_per_aisle <= 4:
#                 print(
#                     f"\nStarting server with {num_aisles} aisles and {units_per_aisle} units per aisle"
#                 )
#                 start_server(num_aisles, units_per_aisle)
#                 break
#             else:
#                 print(
#                     "Invalid values. Number of aisles must be 1-64 and units per aisle must be 1-4."
#                 )
#         except ValueError:
#             print("Please enter numeric values.")
#         except KeyboardInterrupt:
#             print("\nExiting program...")
#             break











########### MY script ................ ####################################################################################################################



# import socket
# import datetime

# HOST = '192.168.1.50'
# PORT = 8235

# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((HOST, PORT))
# server_socket.listen(2)
# print('Waiting for client connection...')
# client_socket, addr = server_socket.accept()
# print('Connecting to client {0}'.format(addr))


# def decode_weight(weight_bytes):
#     weight = weight_bytes[0] + (weight_bytes[1] << 8) + (weight_bytes[2] << 16) + (weight_bytes[3] << 24)
#     return weight * 0.1

# def process_single_client(client_socket, command):
#     try:
        
#         response = bytes(command)
#         client_socket.sendall(response)
       
#         msg = client_socket.recv(1024)
#         if msg:
            
#             if command == [0x02, 0x01, 0x05, 0x00, 0x57, 0x50, 0x05, 0x04, 0x01, 0xF3, 0x03] or command == [0x02, 0x01, 0x05, 0x00, 0x57, 0x50, 0x05, 0x02, 0x02, 0xB6, 0x03]:
#                 print("aisle setup successfully!")

#             elif command == [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]:
#                 print("Tare successfu")

#             else:
#                 hex_a = list(msg)
#                 data_section = hex_a[8:-2]
#                 # Extract weights for four sensors (each weight is 4 bytes)
#                 weight0 = decode_weight(data_section[0:4])
#                 weight1 = decode_weight(data_section[4:8])
#                 weight2 = decode_weight(data_section[8:12])
#                 weight3 = decode_weight(data_section[12:16])

#                 print(f'Received weights -> Weight 0: {weight0}, Weight 1: {weight1}, Weight 2: {weight2}, Weight 3: {weight3}')
#                 # print(f'Received weights -> Weight 0: {weight0}, Weight 1: {weight1}')


#     except Exception as e:
#         print("Error: ", e)
#         # client_socket.close()
            
#     # finally:
#     #     # Close the connection
#     #     client_socket.close()
#     #     server_socket.close()



# # Total we have four sensors

# # set each four aisle id with one weight sensor per lane
# aisle_id = [0x02, 0x01, 0x05, 0x00, 0x57, 0x50, 0x05, 0x04, 0x01, 0xF3, 0x03]

# # set up two aisle id with two weight sensor per lane

# aisle_id_2 = [0x02, 0x01, 0x05, 0x00, 0x57, 0x50, 0x05, 0x02, 0x02, 0xB6, 0x03]


# process_single_client(client_socket, aisle_id)



# # for taring
# TARE = [0x02, 0x01, 0x04, 0x00, 0x57, 0x50, 0x06, 0x00, 0xB2, 0x03]
# process_single_client(client_socket, TARE)


# # get data
# hex_array = [0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03]

# while True:
#     process_single_client(client_socket, hex_array)










#########################################################################################################################################################


# my dynamic scripts




import socket
import datetime

HOST = '192.168.1.220'
PORT = 8234

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(2)
print('Waiting for client connection...')
client_socket, addr = server_socket.accept()
print('Connecting to client {0}'.format(addr))


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


def process_single_client(client_socket, numAisles):
    try:
        response = bytes([0x02, 0x01, 0x04, 0x00, 0x51, 0x50, 0x06, 0x00, 0xac, 0x03])
        client_socket.sendall(response)
        msg = client_socket.recv(1024)

        if msg:
            hex_a = list(msg)
            data_section = hex_a[8:-2]

            sensor_weights = []
            for i in range(numAisles):
                start = i * 4
                end = start + 4
                sensor_weights.append(decode_weight(data_section[start:end]))

            return sensor_weights
            
    except Exception as e:
        print("Error: ", e)
        return []
        # client_socket.close()
            

aisleId = 3
sensorsPerLane = 2
res_1 = setup_aisle_configuration(client_socket, aisleId, sensorsPerLane)
print(f"{res_1} ----> aisle configuration")

res_2 = tare_sensor(client_socket)
print(f"{res_2} -----> Tare")

while True:
    print(process_single_client(client_socket, aisleId))


