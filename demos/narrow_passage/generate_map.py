import matplotlib.pyplot as plt
import numpy as np

from map2d.map2d import Map2D, make_wall
from map2d.robot import Robot
from scipy.interpolate import interp1d

rndst = np.random.RandomState(0)
BUFFER = 1

def one_passage(y, width, length, map_width=30, map_height=30,
                x_left=None, start=None, goal=None, draw=False):
    headless = not draw
    map2d = Map2D(map_width, map_height, headless=headless)

    if x_left is None:
        x_left = -length/2

    y_1 = y + width/2
    y_2 = y - width/2
    wall_top = make_wall(map2d, (x_left+length/2, (map_height/2+y_1)/2),
                         length, map_height/2-y_1)
    wall_bot = make_wall(map2d, (x_left+length/2, (-map_height/2+y_2)/2),
                         length, y_2+map_height/2)

    # sample start and goal
    if start is None:
        start = (rndst.uniform(-map_width/2, x_left-BUFFER), rndst.uniform(-map_height/2, map_height/2))
    if goal is None:
        goal = (rndst.uniform(x_left+length+BUFFER, map_width/2), rndst.uniform(-map_height/2, map_height/2))
    map2d.robot = Robot(map2d, robot_type='point', init_pos=start)

    # generate a solution of the map
    waypoints = [start, (x_left-0.5, y),( x_left+length+0.5, y), goal]
    xs = np.asarray([pnt[0] for pnt in waypoints])
    ys = np.asarray([pnt[1] for pnt in waypoints])
    traj = interp1d(xs, ys, kind='linear')
    xs = np.linspace(start[0], goal[0], num=10, endpoint=True)
    latent_err = rndst.normal(0, 0.1, len(xs))
    ys = map(lambda x,e: (traj(x) + e), xs, latent_err)
    sol = list(zip(xs, ys))
    return map2d, start, goal, sol


if __name__ == '__main__':
    map2d, start, goal, sol = one_passage(0, 2, 10, draw=True)
    while True:
        map2d.step()
