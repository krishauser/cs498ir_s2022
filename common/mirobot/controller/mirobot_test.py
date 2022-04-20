from mirobot import Mirobot

with Mirobot(portname='/dev/ttyUSB0', debug=True) as m:
    print("Homing")
    m.home_individual()
    print("go_to_zero")
    m.go_to_zero()

print("Done.")