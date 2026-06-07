import subprocess

# Typical DJI / Tello MAC prefixes (Organizationally Unique Identifiers)
TELLO_MAC_PREFIXES = [
    "48:1c:b9",  # DJI Technology (Seen in your image)
    "60:60:1f",  # Ryze Tech (Sometimes used by Tellos)
]

def find_tello_drones():
    """
    Checks the local Linux ARP table for MAC addresses matching Tello drones.
    """
    try:
        # Get the ARP table using 'ip neigh'
        output = subprocess.check_output(["ip", "neigh"]).decode("utf-8")
    except Exception as e:
        print(f"Error checking ARP table: {e}")
        return []

    tello_ips = []
    
    # Parse the output
    # Example line: 192.168.0.102 dev wlp0s20f3 lladdr 48:1c:b9:9a:5a:7d REACHABLE
    for line in output.split('\n'):
        parts = line.split()
        if len(parts) >= 5 and "lladdr" in parts:
            ip = parts[0]
            mac_index = parts.index("lladdr") + 1
            mac = parts[mac_index].lower()
            
            # Check if the MAC starts with any known Tello prefix
            for prefix in TELLO_MAC_PREFIXES:
                if mac.startswith(prefix):
                    tello_ips.append((ip, mac))
                    break
                    
    return tello_ips

def ping_subnet(subnet="192.168.0.0/24"):
    """
    Optional: Ping the subnet using nmap to populate the ARP cache before checking.
    Requires nmap to be installed.
    """
    print(f"Pinging {subnet} to discover devices...")
    try:
        subprocess.run(["nmap", "-sn", subnet], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("nmap is not installed. Skipping network ping.")

if __name__ == "__main__":
    # You might want to change this to your actual subnet (e.g., "192.168.1.0/24" if you change routers)
    ping_subnet("192.168.0.0/24")
    
    print("\nChecking for Tello drones...")
    drones = find_tello_drones()
    
    if drones:
        print(f"Found {len(drones)} Tello drone(s) on the network!")
        for i, (ip, mac) in enumerate(drones, start=1):
            print(f"  [Drone {i}] IP: {ip}  |  MAC: {mac}")
    else:
        print("No Tello drones found! Ensure they are powered on and connected to the router.")
