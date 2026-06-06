import socket
import time

TELLO_IP = "192.168.10.1"
TELLO_PORT = 8889

SSID = "TELLO_DEMO"
PASSWORD = "tello12345"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(5)

def send(cmd: str):
    print(f">>> {cmd}")
    sock.sendto(cmd.encode("utf-8"), (TELLO_IP, TELLO_PORT))
    try:
        data, _ = sock.recvfrom(1024)
        print("<<<", data.decode("utf-8"))
    except socket.timeout:
        print("No response (timeout)")

# enter SDK mode first
send("command")
time.sleep(1)

# set station / ap mode: "ap SSID PASSWORD"
send(f"ap {SSID} {PASSWORD}")
time.sleep(1)

sock.close()