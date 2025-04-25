# Load the MuJoCo model
import mujoco
import glfw
import time

if not glfw.init():
    raise Exception("GLFW initialization failed")

model = mujoco.MjModel.from_xml_path("models/my_car.xml")  # Save your XML as car.xml
data = mujoco.MjData(model)

# Create a window
window = glfw.create_window(1200, 900, "MuJoCo Car Control", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

# Make the window's context current
glfw.make_context_current(window)

# Find control indices
forward_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_throttle_velocity")
turn_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "buddy_steering_pos")

# Initialize visualization objects
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")

camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
# Use the camera parameters from the XML file
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = cam_id

# Create scene and context
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# Set initial position
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.05  # Set z-position slightly above ground

# Control parameters
forward_control = 0.0
turn_control = 0.0
speed_factor = 2.0
paused = False
last_time = time.time()


# Key state dictionary
key_state = {
    glfw.KEY_UP: False,
    glfw.KEY_DOWN: False,
    glfw.KEY_LEFT: False,
    glfw.KEY_RIGHT: False
}

# Keyboard callback function
def keyboard_callback(window, key, scancode, action, mods):
    global paused, key_state

    # Update key state dictionary based on press/release
    if key in key_state:
        key_state[key] = (action == glfw.PRESS or action == glfw.REPEAT)

    # Handle single press actions
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            paused = not paused


glfw.set_key_callback(window, keyboard_callback)

print("Control the car with the following keys:")
print("Arrow Up/Down: Forward/Backward")
print("Arrow Left/Right: Turn Left/Right")
print("Space: Pause/resume simulation")
print("Esc: Quit")
print("Mouse: Left click and drag to rotate camera")
print("Mouse: Right click and drag to move camera target")
print("Mouse: Middle click and drag or scroll wheel to zoom")

# Main simulation loop
while not glfw.window_should_close(window):
    # Poll for events
    glfw.poll_events()

    # Get current position of cursor
    last_x, last_y = glfw.get_cursor_pos(window)

    # Calculate time difference
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Update control values based on key states
    forward_control = 0.0
    if key_state[glfw.KEY_UP]:
        forward_control = 1.0 * speed_factor
    elif key_state[glfw.KEY_DOWN]:
        forward_control = -1.0 * speed_factor

    turn_control = 0.0
    if key_state[glfw.KEY_LEFT]:
        turn_control = 1.0 * speed_factor
    elif key_state[glfw.KEY_RIGHT]:
        turn_control = -1.0 * speed_factor

    # Apply controls to the actuators
    data.ctrl[forward_idx] = forward_control
    data.ctrl[turn_idx] = turn_control
    crashed = False
    for i in range(data.ncon):
        contact = data.contact[i]
        # Get geom names
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom1_name is not None and geom2_name is not None:
            if 'buddy' in geom1_name and ('wall' in geom2_name or 'obstacle' in geom2_name) \
            or 'buddy' in geom2_name and ('wall' in geom1_name or 'obstacle' in geom1_name):
                crashed = True
                print(f'Crashed into {geom2_name}, {geom1_name}')
                breakpoint()

    range_sensors = data.sensordata[9:]
    if crashed:
        break
    # Step the simulation
    if not paused:
        mujoco.mj_step(model, data)

    # Get framebuffer size
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(),
        None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene
    )

    # Render scene
    mujoco.mjr_render(viewport, scene, context)

    # Swap front and back buffers
    glfw.swap_buffers(window)
glfw.terminate()
