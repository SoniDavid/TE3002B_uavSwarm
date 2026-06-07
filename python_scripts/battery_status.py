from djitellopy import Tello


if __name__ == "__main__":
	# tello = Tello(host='192.168.0.100')
	tello = Tello()


	try:
		tello.connect()
		battery_info = tello.get_battery()
		print("Drone battery soc: {0}".format(battery_info))
	finally:
		# Ensure sockets/resources are released even if connection fails.
		tello.end()
