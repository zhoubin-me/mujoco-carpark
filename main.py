import mujoco
import numpy as np
import glfw
import time

# Initialize GLFW for window creation and keyboard handling
if not glfw.init():
    raise Exception("GLFW initialization failed")

# Load the MuJoCo model
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
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.distance = 1.5
camera.elevation = -20
camera.azimuth = 90

# Create scene and context
scene = mujoco.MjvScene(model, maxgeom=1000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# Set initial position
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.05  # Set z-position slightly above ground

# Control parameters
forward_control = 0.0
turn_control = 0.0
speed_factor = 1.0
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

# Set the keyboard callback
glfw.set_key_callback(window, keyboard_callback)

# Mouse button callback for camera control
def mouse_button_callback(window, button, act, mods):
    # Store the button state for potential drag operations
    global button_left, button_right, button_middle
    button_left = (button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS)
    button_right = (button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS)
    button_middle = (button == glfw.MOUSE_BUTTON_MIDDLE and act == glfw.PRESS)

# Set the mouse button callback
glfw.set_mouse_button_callback(window, mouse_button_callback)
button_left = False
button_right = False
button_middle = False

# Mouse move callback for camera control
def mouse_move_callback(window, xpos, ypos):
    # Update the camera based on mouse movement
    global button_left, button_right, button_middle, last_x, last_y

    if button_left:
        # Rotate camera
        camera.azimuth += 0.3 * (xpos - last_x)
        camera.elevation += 0.3 * (ypos - last_y)

    if button_right:
        # Move camera target
        forward = np.array([np.cos(np.radians(camera.azimuth)) * np.cos(np.radians(camera.elevation)),
                           np.sin(np.radians(camera.azimuth)) * np.cos(np.radians(camera.elevation)),
                           np.sin(np.radians(camera.elevation))])
        right = np.array([-np.sin(np.radians(camera.azimuth)),
                          np.cos(np.radians(camera.azimuth)),
                          0])
        up = np.cross(right, forward)

        camera.lookat[0] += 0.001 * camera.distance * (xpos - last_x) * right[0]
        camera.lookat[1] += 0.001 * camera.distance * (xpos - last_x) * right[1]
        camera.lookat[2] += 0.001 * camera.distance * (xpos - last_x) * right[2]

        camera.lookat[0] += 0.001 * camera.distance * (ypos - last_y) * up[0]
        camera.lookat[1] += 0.001 * camera.distance * (ypos - last_y) * up[1]
        camera.lookat[2] += 0.001 * camera.distance * (ypos - last_y) * up[2]

    if button_middle:
        # Zoom camera
        camera.distance *= 1.0 + 0.01 * (ypos - last_y)
        if camera.distance < 0.1:
            camera.distance = 0.1

    last_x = xpos
    last_y = ypos

# Set the mouse move callback
glfw.set_cursor_pos_callback(window, mouse_move_callback)
last_x = 0
last_y = 0

# Scroll callback for camera zoom
def scroll_callback(window, xoffset, yoffset):
    global camera
    camera.distance *= 0.9 if yoffset > 0 else 1.1
    if camera.distance < 0.1:
        camera.distance = 0.1

# Set the scroll callback
glfw.set_scroll_callback(window, scroll_callback)

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
    data.ctrl[forward_idx] = 2.0
    data.ctrl[turn_idx] = -2.0

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

# Clean up
glfw.terminate()
