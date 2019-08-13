from Box2D import b2Vec2, b2_pi

# Constants of world
GRAVITY = b2Vec2(0.0, 0.0) # gravity of the world, no gravity here
PPM = 10.0  # pixels per meter
TARGET_FPS = 60 # frame per second
TIME_STEP = 1.0 / TARGET_FPS

VEL_ITERS = 10  # velocity iterations
POS_ITERS = 10  # position iterations

COPY_IGNORE = ('robot')

# Constants of display and keyboard control
KEY_ANGLE = 5/180*b2_pi
OVERCLOCK = 200  # for headless mode only
