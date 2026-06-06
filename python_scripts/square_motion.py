from djitellopy import Tello


BATTERY_MIN_PERCENT = 10
TAKEOFF_HEIGHT_CM = 50
SQUARE_SIDE_CM = 50
SQUARE_TURNS = 4


def fly_square(tello: Tello, side_cm: int) -> None:
	for edge in range(1, SQUARE_TURNS + 1):
		print("Edge {0}/{1}: forward {2} cm".format(edge, SQUARE_TURNS, side_cm))
		tello.move_forward(side_cm)
		tello.rotate_clockwise(90)


if __name__ == "__main__":
	tello = Tello(host='192.168.0.100')
	is_airborne = False

	try:
		tello.connect()

		battery_info = tello.get_battery()
		print("Drone battery soc: {0}%".format(battery_info))

		if battery_info < BATTERY_MIN_PERCENT:
			print(
				"Battery below {0}%. Takeoff aborted for safety.".format(
					BATTERY_MIN_PERCENT
				)
			)
		else:
			tello.takeoff()
			is_airborne = True

			tello.move_up(TAKEOFF_HEIGHT_CM)
			fly_square(tello, SQUARE_SIDE_CM)
			tello.move_down(TAKEOFF_HEIGHT_CM)

			tello.land()
			is_airborne = False

	finally:
		if is_airborne:
			# If any command fails mid-flight, still try to land safely.
			try:
				tello.land()
			except Exception:
				pass
		tello.end()
