# import socket
# import time
# import pickle

# # Configuration
# HOST = "0.0.0.0"  # Listen on all interfaces
# PORT = 8077  # Port for communication with the computer

# LOCAL_HOST_IP = "192.168.1.220"  # Weight sensor IP
# WEIGHT_SENSOR_PORT_1 = 8235  # Weight sensor port
# WEIGHT_SENSOR_PORT_2 = 8234

# def setup_server_socket(ip, port, max_connections=1):
#     """Set up a server socket for listening."""
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     server_socket.bind((ip, port))
#     server_socket.listen(max_connections)
#     return server_socket

# def handle_connections():
#     """Continuously handle connections for the client and weight sensor."""
#     while True:
#         try:
#             print("Setting up 1 weight sensor server...")
#             weight_sensor_socket_1 = setup_server_socket(LOCAL_HOST_IP, WEIGHT_SENSOR_PORT_1)
#             print(f"Weight sensor 1 server listening at {LOCAL_HOST_IP}:{WEIGHT_SENSOR_PORT_1}")

#             print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#             print("Setting up 2 weight sensor server...")
#             weight_sensor_socket_2 = setup_server_socket(LOCAL_HOST_IP, WEIGHT_SENSOR_PORT_2)
#             print(f"Weight sensor 2 server listening at {LOCAL_HOST_IP}:{WEIGHT_SENSOR_PORT_2}")

#             print("Setting up client server...")
#             client_server_socket = setup_server_socket(HOST, PORT)
#             print(f"Client server listening at {HOST}:{PORT}")

#             weight_client_socket_1 = None
#             weight_client_socket_2 = None
#             client_socket = None

#             while True:
#                 # Accept connections for the weight sensor if not connected
#                 if not weight_client_socket_1:
#                     print("Waiting for weight sensor 1 connection...")
#                     weight_client_socket_1, weight_addr_1 = weight_sensor_socket_1.accept()
#                     print(f"Connected to weight sensor 1 at {weight_addr_1}")

#                 if not weight_client_socket_2:
#                     print("Waiting for weight sensor 2 connection...")
#                     weight_client_socket_2, weight_addr_2 = weight_sensor_socket_2.accept()
#                     print(f"Connected to weight sensor 2 at {weight_addr_2}")



#                 # Accept connections for the client if not connected
#                 if not client_socket:
#                     print("Waiting for client connection...")
#                     client_socket, client_addr = client_server_socket.accept()
#                     print(f"Connected to client at {client_addr}")

#                 # If both connections are established, handle communication
#                 if client_socket and weight_client_socket_1 and weight_client_socket_2:
#                     try:
#                         # Receive command from the client
#                         command = client_socket.recv(1024)
#                         if not command:
#                             print("Client disconnected.")
#                             client_socket.close()
#                             client_socket = None
#                             continue

#                         # print(f"Received command from client: {command.decode()}")

#                         # Send the command to the weight sensor

#                         command_data = pickle.loads(command)

#                         weight_client_socket_1.sendall(bytes(command_data[0]))
#                         weight_client_socket_2.sendall(bytes(command_data[1]))

#                         # Get response from the weight sensor
#                         response_1 = weight_client_socket_1.recv(1024)
#                         response_2 = weight_client_socket_2.recv(1024)
                        
#                         # Send the response back to the client
#                         client_socket.sendall(pickle.dumps([response_1, response_2]))
#                         # print(f"Response sent back to client: {response.decode()}")

#                     except Exception as e:
#                         print(f"Error during communication: {e}")
#                         if client_socket:
#                             client_socket.close()
#                             client_socket = None

#                         if weight_client_socket_1:
#                             weight_client_socket_1.close()
#                             weight_client_socket_1 = None

#                         if weight_client_socket_2:
#                             weight_client_socket_2.close()
#                             weight_client_socket_2 = None
                
#         except Exception as e:
#             print(f"Error in server setup or communication: {e}")
#             time.sleep(1)  # Short delay before retrying

# if __name__ == "__main__":
#     handle_connections()




########## dynamic script for multi weight processors"


import socket
import time
import pickle

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8077  # Port for communication with the computer
LOCAL_HOST_IP = "59.103.120.15"  # LOCAL_HOST_IP (Remote ip)
WEIGHT_SENSOR_PORTS = [8234, 8235]  # List of weight sensor ports

def setup_server_socket(ip, port, max_connections=1):
    """Set up a server socket for listening."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(max_connections)
    return server_socket

def handle_connections():
    """Continuously handle connections for the client and weight sensors."""
    while True:
        try:
            # Setup weight sensor servers dynamically
            weight_sensor_sockets = {}
            for port in WEIGHT_SENSOR_PORTS:
                print(f"Setting up weight sensor server at port {port}...")
                weight_sensor_sockets[port] = setup_server_socket(LOCAL_HOST_IP, port)
                print(f"Weight sensor server listening at {LOCAL_HOST_IP}:{port}")
            
            print("Setting up client server...")
            client_server_socket = setup_server_socket(HOST, PORT)
            print(f"Client server listening at {HOST}:{PORT}")

            weight_client_sockets = {port: None for port in WEIGHT_SENSOR_PORTS}
            client_socket = None

            while True:
                # Accept connections for the weight sensors if not connected
                for port in WEIGHT_SENSOR_PORTS:
                    if not weight_client_sockets[port]:
                        print(f"Waiting for weight sensor connection at port {port}...")
                        weight_client_sockets[port], weight_addr = weight_sensor_sockets[port].accept()
                        print(f"Connected to weight sensor at {weight_addr}")

                # Accept connection for the client if not connected
                if not client_socket:
                    print("Waiting for client connection...")
                    client_socket, client_addr = client_server_socket.accept()
                    print(f"Connected to client at {client_addr}")

                # If all connections are established, handle communication
                if client_socket and all(weight_client_sockets.values()):
                    try:
                        # Receive command from the client
                        command = client_socket.recv(1024)
                        if not command:
                            print("Client disconnected.")
                            client_socket.close()
                            client_socket = None
                            continue

                        # Decode the command
                        command_data = pickle.loads(command)
                        
                        # Send commands to all weight sensors
                        responses = []
                        for i, port in enumerate(WEIGHT_SENSOR_PORTS):
                            weight_client_sockets[port].sendall(bytes(command_data[i]))
                            responses.append(weight_client_sockets[port].recv(1024))

                        # Send responses back to the client
                        client_socket.sendall(pickle.dumps(responses))

                    except Exception as e:
                        print(f"Error during communication: {e}")
                        if client_socket:
                            client_socket.close()
                            client_socket = None

                        for port in WEIGHT_SENSOR_PORTS:
                            if weight_client_sockets[port]:
                                weight_client_sockets[port].close()
                                weight_client_sockets[port] = None
                
        except Exception as e:
            print(f"Error in server setup or communication: {e}")
            time.sleep(1)  # Short delay before retrying

if __name__ == "__main__":
    handle_connections()
