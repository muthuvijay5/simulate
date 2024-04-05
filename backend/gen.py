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


def ic(r1x1, r1y1, r1x2, r1y2, r2x1, r2y1, r2x2, r2y2):
    if r1x2 < r2x1 or r2x2 < r1x1:
        return False
    if r1y2 < r2y1 or r2y2 < r1y1:
        return False
    return True


def ics(cv, vp):
    for i in vp:
        if ic(i[0], i[1], i[2], i[3], cv[0], cv[1], cv[2], cv[3]):
            return True
    return False


veh = [i for i in range(1, 31)]
obs = [i for i in range(11)]
lane = [i for i in range(3, 21)]
speeds = [i for i in range(151)]
pos = [i for i in range(2, 301)]

for _ in range(30000):
    l = random.choice(lane)
    nv = random.choice(veh)
    cv = [random.choice([i for i in range(l - 2)]), random.choice(pos)]
    cv.append(cv[0] + 2)
    cv.append(cv[1] - 2)
    cvs = random.choice(speeds)

    vp = []
    for __ in range(nv - 1):
        x = [random.choice([i for i in range(l - 2)]), random.choice(pos)]
        x.append(x[0] + 2)
        x.append(x[1] - 2)
        while ic(x[0], x[1], x[2], x[3], cv[0], cv[1], cv[2], cv[3]) or ics(x, vp):
            x = [random.choice([i for i in range(l - 2)]), random.choice(pos)]
            x.append(x[0] + 2)
            x.append(x[1] - 2)
        vp.append(x)

    s = [random.choice(speeds) for i in range(nv - 1)]

    o = []
    for __ in range(random.choice(obs)):
        x = [random.choice([i for i in range(l)]), random.choice(pos)]
        while (
            ic(x[0], x[1], x[0], x[1], cv[0], cv[1], cv[2], cv[3])
            or ics(x + x, vp)
            or x in o
        ):
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

# veh = [20]
# obs = [5]
# lane = [5]
# speeds = [i for i in range(151)]
# pos = [i for i in range(201)]

# for _ in range(30000):
#     l = random.choice(lane)
#     nv = random.choice(veh)
#     cv = [random.choice([i for i in range(l)]), random.choice(pos)]
#     cvs = random.choice(speeds)

#     vp = []
#     for __ in range(nv - 1):
#         x = [random.choice([i for i in range(l)]), random.choice(pos)]
#         while x == cv or x in vp:
#             x = [random.choice([i for i in range(l)]), random.choice(pos)]
#         vp.append(x)

#     s = [random.choice(speeds) for i in range(nv - 1)]

#     o = []
#     for __ in range(random.choice(obs)):
#         x = [random.choice([i for i in range(l)]), random.choice(pos)]
#         while x == cv or x in vp or x in o:
#             x = [random.choice([i for i in range(l)]), random.choice(pos)]
#         o.append(x)

#     tq = [random.random() for i in range(nv - 1)]

#     num_lane.append(str(l))
#     num_vehicles.append(str(nv))
#     current_vehicle.append(str(cv))
#     current_vehicle_speed.append(str(cvs))
#     vehicle_positions.append(str(vp))
#     speed.append(str(s))
#     obstacle_positions.append(str(o))
#     transmission_quality.append(str(tq))

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
