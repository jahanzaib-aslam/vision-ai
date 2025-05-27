import socket
import time

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8087  # Port for communication with the computer

WEIGHT_SENSOR_IP = "192.168.1.142"  # LOCAL_HOST_IP
WEIGHT_SENSOR_PORT = 8235  # Weight sensor port

def setup_server_socket(ip, port, max_connections=1):
    """Set up a server socket for listening."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(max_connections)
    return server_socket

def handle_connections():
    """Continuously handle connections for the client and weight sensor."""
    while True:
        try:
            print("Setting up weight sensor server...")
            weight_sensor_socket = setup_server_socket(WEIGHT_SENSOR_IP, WEIGHT_SENSOR_PORT)
            print(f"Weight sensor server listening at {WEIGHT_SENSOR_IP}:{WEIGHT_SENSOR_PORT}")

            print("Setting up client server...")
            client_server_socket = setup_server_socket(HOST, PORT)
            print(f"Client server listening at {HOST}:{PORT}")

            weight_client_socket = None
            client_socket = None

            while True:
                # Accept connections for the weight sensor if not connected
                if not weight_client_socket:
                    print("Waiting for weight sensor connection...")
                    weight_client_socket, weight_addr = weight_sensor_socket.accept()
                    print(f"Connected to weight sensor at {weight_addr}")

                # Accept connections for the client if not connected
                if not client_socket:
                    print("Waiting for client connection...")
                    client_socket, client_addr = client_server_socket.accept()
                    print(f"Connected to client at {client_addr}")

                # If both connections are established, handle communication
                if client_socket and weight_client_socket:
                    try:
                        # Receive command from the client
                        command = client_socket.recv(1024)
                        if not command:
                            print("Client disconnected.")
                            client_socket.close()
                            client_socket = None
                            continue

                        # print(f"Received command from client: {command.decode()}")

                        # Send the command to the weight sensor
                        weight_client_socket.sendall(command)

                        # Get response from the weight sensor
                        response = weight_client_socket.recv(1024)

                        # Send the response back to the client
                        client_socket.sendall(response)
                        # print(f"Response sent back to client: {response.decode()}")

                    except Exception as e:
                        print(f"Error during communication: {e}")
                        if client_socket:
                            client_socket.close()
                            client_socket = None
                        if weight_client_socket:
                            weight_client_socket.close()
                            weight_client_socket = None
                
        except Exception as e:
            print(f"Error in server setup or communication: {e}")
            time.sleep(1)  # Short delay before retrying

if __name__ == "__main__":
    handle_connections()