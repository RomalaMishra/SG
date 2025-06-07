import serial.tools.list_ports
import pyttsx3

def say_warning(identifier, distance):
    engine = pyttsx3.init()
    if distance < 30:
        if identifier == "Distance 1":
            engine.say(f"Careful! There is an object ahead at distance {distance} centimeters.")
        elif identifier == "Distance 2":
            engine.say(f"Careful! There is an object to the left at distance {distance} centimeters.")
        elif identifier == "Distance 3":
            engine.say(f"Careful! There is an object to the right at distance {distance} centimeters.")
        engine.runAndWait()

ports = serial.tools.list_ports.comports()
portsList = [str(onePort) for onePort in ports]

for onePort in portsList:
    print(onePort)

val = input("Select Port: COM")

for port in portsList:
    if port.startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)

serialInst = serial.Serial(portVar, 9600)

while True:
    if serialInst.in_waiting:
        packet = serialInst.readline().decode().strip()  # Read serial data and remove leading/trailing whitespaces

        # Split the packet by ":" and get the first part (identifier) and second part (distance value)
        parts = packet.split(":")
        if len(parts) >= 2:
            identifier = parts[0].strip()
            distance_str = parts[1].strip()

            try:
                distance = float(distance_str)  # Convert distance string to float
            except ValueError:
                print(f"Error converting distance: {distance_str}")
                continue

            print(f"{identifier}: {distance} cm")

            # Check if the distance is below 30 for each identifier and say warning if true
            say_warning(identifier, distance)
