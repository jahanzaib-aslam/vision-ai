import socketio
import logging
from threading import Thread, Event
import json
import time


# Create a Socket.IO client
sio = socketio.Client()

# Event to notify when connection is established
connection_event = Event()


socket_url = 'https://socket.unlimitretail.dk:3000' # Staging

# socket_url = 'https://socket.dev.unlimitretail.dk:3000' # Dev


# Function to connect to the server
def connect_to_server():
    try:
        sio.connect(socket_url)
        print("Connected to server")
        connection_event.set()  # Set event when connected
    except socketio.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")

# Function to ensure the connection is established
def ensure_connection():
    if not sio.connected:
        connect_to_server()
        connection_event.wait()  # Wait until connection is established

# Function to send data on "pickup" event
def send_pickup_event(event, user_id, data):
    ensure_connection() # Ensure the connection is established
    data = {
        'data': json.dumps(data),
        'user_id': user_id
    }
    sio.emit(event, data)
    print(f"Sent {event} event with data: {data}")


# Function to close the connection
def close_connection():
    sio.emit('close_connection')  # Emit a custom event to close the connection
    sio.disconnect()  # Disconnect the Socket.IO client
    print("Connection closed")

# Ensure the script runs the connection in a separate thread
if __name__ == "__main__":
    connect_thread = Thread(target=connect_to_server)
    connect_thread.start()
    # Wait for the connection to be established before continuing
    connection_event.wait()
    sio.wait()
