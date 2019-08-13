from Box2D import *
from Box2D.b2 import *
from map2d.constants import *
from map2d.framework import b2WorldInterface
from map2d.robot import Robot

import json
import numpy as np


def copy_map(map2d):
    '''
    Make a copy of map with obstacles for collision checking.
    '''
    b2w = Map2D(map2d.width, map2d.height, headless=True,
    	        overclock=map2d.overclock, caption='Buffer')
    for b in map2d.world.bodies:
        name = b.userData['name']
        if name in COPY_IGNORE:
            continue
        pose = np.hstack((b.position, b.angle))
        user_data = b.userData.copy()
        body = make_body(map2d=b2w, name=name, pose=pose, args=user_data)
    return b2w


class Map2D(b2WorldInterface):
    '''
    Origin of the coordinate is at the center of the map
    '''
    def __init__(self, width, height, headless=False, overclock=None, caption=''):
        if headless: overclock = OVERCLOCK
        super(Map2D, self).__init__(width, height, True, headless,
        	                        overclock=overclock, caption=caption)
        self.robot = None
        self.overclock = overclock
        self.width = width
        self.height = height

    def draw(self):
        if self.gui_world is not None and self.robot is not None:
            self.gui_world.move(self.robot, self)
        super(Map2D, self).draw()

    def has_contact(self):
        return len(self.robot.contacts) > 0

    def is_obstacle_free(self, state):
        point = state.value[:2]
        for b in self.world.bodies:
            name = b.userData['name']
            if 'obs_' not in name:
                continue
            else:
                for fixture in b.fixtures:
                    if fixture.TestPoint(point):
                        return False
        return True

    def check_collision(self, pos, angle, return_map=False):
        new_map = copy_map(self)
        robot = Robot(new_map,
                      init_pos=self.robot.position,
                      init_angle=self.robot.angle)
        new_map.robot = robot
        _, collide = robot.apply_lowlevel_control(pos, angle, collision_check=True)
        new_map.step(); new_map.draw()
        if return_map:
            return collide, new_map
        else:
            return collide

    def save_json(self, filename, parse=None):
        map_data = {'objects': []}
        for b in self.world.bodies:
            user_data = b.userData.copy()
            if (user_data['name'] in COPY_IGNORE):
                continue
            user_data['position'] = (b.position[0], b.position[1])
            user_data['angle'] = b.angle
            map_data['objects'].append(user_data)
        map_data['robot'] = {
            'init_pos': (self.gripper.position[0], self.gripper.position[1]),
            'init_angle': self.gripper.angle
        }
        with open(filename, 'w') as fp:
            json.dump(map_data, fp, indent=4)

    @staticmethod
    def load_json(filename):
        data = json.load(open(filename))
        map2d = Map2D()
        for obj in data['objects']:
            pose = np.hstack((obj['position'], obj['angle']))
            obj.pop('position'); obj.pop('angle');
            make_body(map2d, obj['name'], pose, obj)
        map2d.robot = Robot(map2d, np.asarray(data['robot']['init_pos']),
                            data['robot']['init_angle'])
        return map2d


def make_body(map2d, name, pose, args):
    '''
    A wrapper to create Box2d bodies based on their name, pose and size.
    '''
    args.pop('name');
    if 'robot' in name:
        body = Robot(map2d, init_pos=pose[:2], init_angle=pose[2])
    elif name == 'obs_wall':
        body = make_wall(map2d, pose[:2], **args)
    else:
        raise NotImplementedError(name)
    body.name = name
    return body

def make_wall(b2world_interface, pos, w, h, color='wall'):
    world = b2world_interface.world
    body = world.CreateStaticBody(
        position=pos
    )
    polygon_shape = [(w/2, h/2), (w/2, -h/2), (-w/2, -h/2), (-w/2, h/2)]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        friction=1,
        density=0.5,
        restitution=0
    )
    body.userData = {'name': 'obs_wall', 'w': w, 'h': h, 'color': color}
    return body
