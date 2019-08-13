import numpy as np
import map2d.framework

from Box2D import *
from Box2D.b2 import *
from math import cos, sin
from map2d.constants import *

def create_point_robot(world, init_pos, r=0.2, color='red'):
    '''
    Create a robot with a single point.
    Args: 
        init_pos: position of the robot.
        r: radius of the single point robot.
    '''
    body = world.CreateDynamicBody(position=init_pos, allowSleep=True)
    body.CreateFixture(
        shape=b2CircleShape(pos=(0,0), radius=r),
        density=100.,
        friction=10.,
        restitution=0.
    )
    body.userData = {'name': 'robot', 'color': color}
    body.angularDamping = 0.01
    return body

def create_simple_car_robot(world, init_pos, init_angle, w=0.23, h=0.14, color='red'):
    '''
    Create a robot with a simple rectangle.
    Args: 
        init_pos: position of the simple robot.
        init_angle: initial angle of the simple robot.
        w: width.
        h: height.
    '''
    body = world.CreateDynamicBody(
        position=init_pos,
        angle=init_angle
    )
    polygon_shape = [(0, 0), (w, 0), (w, h), (0, h)]
    polygon_shape = [(v[0]-w/2., v[1]-h/2.) for v in polygon_shape]
    body.CreateFixture(
        shape=b2PolygonShape(vertices=polygon_shape),
        density=100.,
        friction=10.,
        restitution=0.
    )
    body.userData = {'name': 'robot', 'color': color}
    body.angularDamping = 0.01
    return body

class Robot(object):
    def __init__(self, b2world_interface,
    	         robot_type='point',
                 init_pos=None,
                 init_angle=0,
                 color='red'):
        world = b2world_interface.world
        self.b2w = b2world_interface

        self.robot_type = robot_type
        if robot_type == 'point':
        	self.body = create_point_robot(world, init_pos, color=color)
        elif robot_type == 'simple_car':
        	self.body = create_simple_car_robot(world, init_pos, init_angle, color=color)
        else:
        	raise NotImplementedError(robot_type)

    @property
    def mass(self):
        return self.body.mass

    @property
    def position(self):
        return self.body.position

    @property
    def angle(self):
        return self.body.angle

    @property
    def inertia(self):
        return self.body.inertia

    @property
    def velocity(self):
        linear_v = self.body.linearVelocity
        angular_v = self.body.angularVelocity
        return np.hstack((linear_v, angular_v))

    def check_collision(self):
        for c in self.body.contacts:
            if c.contact.touching and ('obs_' in c.contact.fixtureA.body.userData['name'] \
                                       or 'obs_' in c.contact.fixtureB.body.userData['name']):
                return True
        return False

    def apply_lowlevel_control(self, dpos, dangle, maxspeed=10., collision_check=False):
        '''
        Apply a lowlevel controller to the robot, which moves to the target pose 
        by following the straight line interpolation from the current pose.
        Args:
            dpos: target position.
            dangle: target angle.
            maxspeed: the maximum speed of moving.
            collision_check: indicator of whether to check collision along the way.
        Returns: 
            if collision_check is True, it returns the trajectory until collision happened;
            otherwise, return the length of the trajectory.
        '''
        dpos = b2Vec2(dpos)
        dposa = np.hstack((dpos, dangle))
        cur_pos = np.hstack((self.position, self.angle))

        t0 = 1 # timesteps for acceleration and deceleration
        assert maxspeed > 0
        max_timesteps = int(np.max(np.abs(dposa - cur_pos)/maxspeed) / TIME_STEP + 1) + t0
        t = max_timesteps

        cur_v = self.velocity
        v = (dposa - cur_pos - cur_v*t0*TIME_STEP/2) / (t-t0) / TIME_STEP
        ddy = np.array([(v-cur_v)/t0/TIME_STEP]*t0 + [[0]*3]*(t-2*t0) + [-v/t0/TIME_STEP]*t0)
        traj = [[self.position[0], self.position[1], self.angle]]
        for step in range(len(ddy)):
            force = (ddy[step,:2]) * self.mass/2
            self.body.ApplyForce(force, self.body.position, wake=True)
            self.b2w.step()
            self.body.angle = dangle
            self.b2w.step()
            traj.append([self.position[0], self.position[1], self.angle])
            if collision_check and self.check_collision():
                return traj, True
        # return trajectory
        if collision_check:
            return traj, False
        else:
            return traj, len(ddy)
