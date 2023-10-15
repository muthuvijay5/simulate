import random
import pandas as pd

dataset = pd.DataFrame()
num_lane = []
num_vehicles = []
current_vehicle = []
current_vehicle_speed = []
vehicle_positions = []
speed = []
obstacle_positions = []
transmission_quality = []

veh = [i for i in range(1, 31)]
obs = [i for i in range(11)]
lane = [i for i in range(1, 7)]
speeds = [i for i in range(151)]
pos = [i for i in range(201)]

for _ in range(10000):
    l = random.choice(lane)
    nv = random.choice(veh)
    cv = [random.choice([i for i in range(l)]), random.choice(pos)]
    cvs = random.choice(speeds)

    vp = []
    for __ in range(nv - 1):
        x = [random.choice([i for i in range(l)]), random.choice(pos)]
        while x == cv or x in vp:
            x = [random.choice([i for i in range(l)]), random.choice(pos)]
        vp.append(x)

    s = [random.choice(speeds) for i in range(nv - 1)]

    o = []
    for __ in range(random.choice(obs)):
        x = [random.choice([i for i in range(l)]), random.choice(pos)]
        while x == cv or x in vp or x in o:
            x = [random.choice([i for i in range(l)]), random.choice(pos)]
        o.append(x)

    tq = [random.random() for i in range(nv - 1)]

    num_lane.append(str(l))
    num_vehicles.append(str(nv))
    current_vehicle.append(str(cv))
    current_vehicle_speed.append(str(cvs))
    vehicle_positions.append(str(vp))
    speed.append(str(s))
    obstacle_positions.append(str(o))
    transmission_quality.append(str(tq))

print(len(num_lane))
print(len(num_vehicles))
print(len(current_vehicle))
print(len(current_vehicle_speed))
print(len(vehicle_positions))
print(len(speed))
print(len(obstacle_positions))
print(len(transmission_quality))

dataset["num_lane"] = num_lane
dataset["num_vehicles"] = num_vehicles
dataset["current_vehicle"] = current_vehicle
dataset["current_vehicle_speed"] = current_vehicle_speed
dataset["vehicle_positions"] = vehicle_positions
dataset["speed"] = speed
dataset["obstacle_positions"] = obstacle_positions
dataset["transmission_quality"] = transmission_quality
dataset.to_csv("dataset.csv")
