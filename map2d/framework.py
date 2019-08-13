from Box2D import *
from Box2D.b2 import *
from map2d.constants import *

import numpy as np
import os
import pygame


class guiWorld:
    def __init__(self, width, height, caption='PyBox2D Simulator', overclock=None):
        '''
        Graphics wrapper for visualization of Pybox2D with pygame. 
        Args:
            width: width of the world in meter.
            height: height of the world in meter.
            caption: caption on the window.
            overclock: number of frames to skip when showing graphics.
        '''
        self.width_px = int(width * PPM)
        self.height_px = int(height * PPM)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width_px, self.height_px), 0, 32)
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.overclock = overclock
        self.screen_origin = b2Vec2(width/2., height/2.)
        self.colors = {
            'obstacle': (123, 128, 120, 255),
            'wall': (155, 155, 155, 255),
            'default': (81, 81, 81, 255),
            'red': (200, 0, 0, 255),
            'green': (0, 99, 0),
            'blue': (255, 0, 0, 0),
            'black': (0, 0, 0),
            'yellow': (244, 170, 66, 255),
            'pink': (255, 192, 203),
            'purple': (128,0,128),
            'orange': (255,140,0)
        }

    def move(self, robot, map2d=None):
        for event in pygame.event.get():
            KEY_POSITION = 1
            KEY_ANGLE = 0.1
            if event.type == pygame.KEYDOWN:
                new_pos = robot.position
                new_angle = robot.angle
                if robot.robot_type == 'point' or robot.robot_type == 'simple_car':
                    if event.key == pygame.K_UP:
                        new_pos = (robot.position[0], robot.position[1]+KEY_POSITION)
                    elif event.key == pygame.K_DOWN:
                        new_pos = (robot.position[0], robot.position[1]-KEY_POSITION)
                    if event.key == pygame.K_RIGHT:
                        new_pos = (robot.position[0]+KEY_POSITION, robot.position[1])
                    elif event.key == pygame.K_LEFT:
                        new_pos = (robot.position[0]-KEY_POSITION, robot.position[1])
                if robot.robot_type == 'simple_car':
                    if event.key == pygame.K_a:
                        new_angle = robot.angle + KEY_ANGLE
                    elif event.key == pygame.K_s:
                        new_angle = robot.angle - KEY_ANGLE
                robot.apply_lowlevel_control(new_pos, new_angle)

    def draw(self, bodies, bg_color=(255, 255, 255, 1)):
        '''
        Draw bodies in the world with pygame.
        Adapted from examples/simple/simple_02.py in pybox2d.
        Args:
            bodies: a list of box2d bodies
            bg_color: background color 
        '''
        def my_draw_polygon(polygon, body, fixture, debug_center=False):
            vertices = [(self.screen_origin + body.transform*v)
                        * PPM for v in polygon.vertices]
            vertices = [(v[0], self.height_px-v[1]) for v in vertices]
            if 'color' in body.userData.keys() and body.userData['color'] is not None:
                color = self.colors.get(body.userData['color'], self.colors['default'])
            else:
                name = body.userData['name'].replace('obs_', '')
                color = self.colors.get(name, self.colors['default'])
            pygame.draw.polygon(self.screen, color, vertices)
            if debug_center: # draw body center to check object coordinates
                position = (self.screen_origin + body.transform.position)*PPM
                position = (position[0], self.height_px-position[1])
                pygame.draw.circle(self.screen, (0,0,0), [int(x) for x in position],
                                   int(0.1*PPM))

        def my_draw_circle(circle, body, fixture):
            position = (self.screen_origin + body.transform*circle.pos)*PPM
            position = (position[0], self.height_px-position[1])
            if 'color' in body.userData.keys() and body.userData['color'] is not None:
                color = self.colors.get(body.userData['color'], self.colors['default'])
            else:
                name = body.userData['name'].replace('obs_', '')
                color = self.colors.get(name, self.colors['default'])
            pygame.draw.circle(self.screen, color, [int(x) for x in position],
                               int(circle.radius*PPM))

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle
        self.screen.fill(bg_color)
        if self.overclock is None:
            self.clock.tick(TARGET_FPS)
        pygame.event.get()
        for body in bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        pygame.display.flip()


class b2WorldInterface(object):
    def __init__(self, width, height, do_gui=True, headless=False,
                 caption='PyBox2D Simulator', overclock=OVERCLOCK):
        '''
        Interface between Pybox2D and the graphics wrapper guiWorld.
        Args:
            width: width of the world in meter.
            height: height of the world in meter.
            do_gui: True if rendering graphics; otherwise False.
            headless: True if don't want to show graphics on the display or running on server.
            caption: caption on the simulator graphics window.
            overclock: number of frames to skip when showing graphics. If overclock is None, 
            this feature is not used.
        '''
        if headless: # enable this for headless mode
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.world = b2World(gravity=GRAVITY, doSleep=True)
        self.do_gui = do_gui
        if do_gui:
            self.gui_world = guiWorld(width, height, caption=caption, overclock=overclock)
        else:
            self.gui_world = None
        self.num_steps = 0
        self.overclock = overclock

    def enable_gui(self, caption='PyBox2D Simulator'):
        '''
        Enable visualization.
        '''
        self.do_gui = True
        self.gui_world = guiWorld(caption=caption, overclock=self.overclock)

    def disable_gui(self):
        '''
        Disable visualization.
        '''
        self.do_gui = False

    def draw(self):
        '''
        Visualize the current scene if do_gui is Trues.
        '''
        if not self.do_gui:
            return
        self.gui_world.draw(self.world.bodies)

    def step(self):
        '''
        Wrapper of the step function of b2World.
        '''
        self.world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()
        self.num_steps += 1
        if (self.overclock is None) or (self.num_steps % self.overclock == 0):
            self.draw()
