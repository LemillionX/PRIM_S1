import pygame
import pygame_gui
import callbacksUI as ui
import numpy as np

#####################################
#           SETTINGS                #
#####################################
GRID_SIZE = 20
WINDOW_RESOLUTION = 900


#####################################
#           INTERFACE               #
#####################################
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0,255,0)

# Grid dimensions
GRID_WIDTH = GRID_SIZE
GRID_HEIGHT = GRID_SIZE
WINDOW_WIDTH = WINDOW_RESOLUTION
WINDOW_HEIGHT = WINDOW_RESOLUTION
WIDGET_HEIGHT = 80
CELL_SIZE = WINDOW_HEIGHT//GRID_HEIGHT
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Button settings
BUTTON_WIDTH = 130
BUTTON_HEIGHT = 20
BUTTON_TOP = 2
BUTTON_LEFT = 5

# Brush settings
LINE_WIDTH = 5
LINE_COLOR = RED

# Initialize Pygame
pygame.init()

# Create the window
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Interactive Grid")

# Create the UI
ui_manager = pygame_gui.UIManager(WINDOW_SIZE)

# Create a tool bar container
tool_bar_container = pygame_gui.elements.UIPanel(
    relative_rect=pygame.Rect(0, 0, WINDOW_SIZE[0], 30),
    starting_layer_height=1,
    manager=ui_manager
)

# Add buttons to the tool bar
load_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(5, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT),
    text='Load',
    manager=ui_manager,
    container=tool_bar_container
)


save_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(BUTTON_WIDTH + 1.5*BUTTON_LEFT, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT),
    text='Save',
    manager=ui_manager,
    container=tool_bar_container
)

reset_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(2*BUTTON_WIDTH + 2*BUTTON_LEFT, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT),
    text='Reset',
    manager=ui_manager,
    container=tool_bar_container
)

edit_density_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(3*BUTTON_WIDTH + 2*BUTTON_LEFT, BUTTON_TOP, BUTTON_WIDTH*1.5, BUTTON_HEIGHT),
    text='Edit Target Density',
    manager=ui_manager,
    container=tool_bar_container
)

edit_init_density_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(4.5*BUTTON_WIDTH + 2*BUTTON_LEFT, BUTTON_TOP, BUTTON_WIDTH*1.5, BUTTON_HEIGHT),
    text='Edit Initial Density',
    manager=ui_manager,
    container=tool_bar_container
)

# Tool bar for target density grid
tool_bar_density = pygame_gui.elements.UIPanel(
    relative_rect=pygame.Rect(0, 0, WINDOW_SIZE[0], 30),
    starting_layer_height=1,
    manager=ui_manager
)

# Tool bar for initial density grid
tool_bar_init_density = pygame_gui.elements.UIPanel(
    relative_rect=pygame.Rect(0, 0, WINDOW_SIZE[0], 30),
    starting_layer_height=1,
    manager=ui_manager
)

# Come back to UI Tool bar button
quit_density_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(5, BUTTON_TOP, BUTTON_WIDTH*1.5, BUTTON_HEIGHT),
    text='End density editing',
    manager=ui_manager,
    container=tool_bar_density
)

# Come back to UI Tool bar button
quit_init_density_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(5, BUTTON_TOP, BUTTON_WIDTH*1.5, BUTTON_HEIGHT),
    text='End density editing',
    manager=ui_manager,
    container=tool_bar_init_density
)


# Set the default background color
screen.fill(WHITE)

# Draw the grid
for x in range(0, WINDOW_WIDTH, CELL_SIZE):
    pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT))
for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
    pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))

# Create a list to store the curves and visited cells
curves = []
visited_cells = []

# Create a grid to store density
target_density = np.zeros(GRID_HEIGHT*GRID_WIDTH)
init_density = np.zeros(GRID_HEIGHT*GRID_WIDTH)


# Active panel variable
active_panel = tool_bar_container
lst_panels = [tool_bar_container, tool_bar_density, tool_bar_init_density]


# Define callbacks for buttons
def save_trajectory():
    file_name = ui.prompt_file()
    if file_name is not None:
        if len(visited_cells) > 0:
            ui.saveToJSON(visited_cells[0], target_density.tolist(), init_density.tolist(), curves, GRID_HEIGHT, file_name)
        else:
            ui.saveToJSON([], target_density.tolist(), init_density.tolist(), GRID_HEIGHT, [], file_name)
        pygame.image.save(screen, file_name.rsplit(".",1)[0] + ".jpg")
        print("Trajectory saved here : ", file_name)

def load_density():
    data = ui.loadFromJSON()
    if data is not None:
        loaded_visited_cells = [[ [x[0]%CELL_SIZE, x[0]//CELL_SIZE] for x in data["indices"]]]
        loaded_target_density = np.array(data["target_density"])
        loaded_init_density = np.array(data["init_density"])
        loaded_curves = []
        if "curves" in data:
            loaded_curves = data["curves"]
    return loaded_visited_cells, loaded_target_density, loaded_init_density, loaded_curves

def reset():
    print("Reset")
    curves.clear()
    visited_cells.clear()


# Boolean to indicate drawing state 
drawing = False

# Start the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and GRID_HEIGHT-1 - event.pos[1]//CELL_SIZE < GRID_HEIGHT-1:
                # If the left mouse button is pressed, start a new curve
                if active_panel == tool_bar_container:
                    drawing = True
                    curves.append([event.pos])
                    visited_cells.append([[event.pos[0]//CELL_SIZE, GRID_HEIGHT-1 - event.pos[1]//CELL_SIZE]])
                if active_panel == tool_bar_density:
                    i = event.pos[0]//CELL_SIZE
                    j = GRID_HEIGHT-1 - event.pos[1]//CELL_SIZE
                    target_density[i+j*GRID_HEIGHT] = 1 - target_density[i+j*GRID_HEIGHT]
                if active_panel == tool_bar_init_density:
                    i = event.pos[0]//CELL_SIZE
                    j = GRID_HEIGHT-1 - event.pos[1]//CELL_SIZE
                    init_density[i+j*GRID_HEIGHT] = 1 - init_density[i+j*GRID_HEIGHT]
                    # print(i + GRID_HEIGHT * j)
        elif event.type == pygame.MOUSEMOTION and drawing:
            # If the left mouse button is pressed and moving, continue the curve
            curves[-1].append(pygame.mouse.get_pos())
            if visited_cells[-1][-1] != [pygame.mouse.get_pos()[0]//CELL_SIZE, GRID_HEIGHT-1 - pygame.mouse.get_pos()[1]//CELL_SIZE]:
                visited_cells[-1].append([pygame.mouse.get_pos()[0]//CELL_SIZE, GRID_HEIGHT-1 - pygame.mouse.get_pos()[1]//CELL_SIZE])
        elif event.type == pygame.MOUSEBUTTONUP:
            # If the left mouse button is released, add the curve to all curves
            if event.button == 1:
                drawing = False
                # print(visited_cells)

        # Check for button press event
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == load_button:
                    visited_cells, target_density, init_density, curves = load_density()
                if event.ui_element == save_button:
                    save_trajectory()
                if event.ui_element == reset_button:
                    reset()
                if event.ui_element == edit_density_button:
                    active_panel = tool_bar_density
                if event.ui_element == edit_init_density_button:
                    active_panel = tool_bar_init_density
                if event.ui_element == quit_density_button:
                    active_panel = tool_bar_container
                if event.ui_element == quit_init_density_button:
                    active_panel = tool_bar_container

        # Update the UI manager with the event
        ui_manager.process_events(event)



    # Redraw the screen
    screen.fill(WHITE)

    # Draw density
    for idx in range(len(target_density)):
        row = idx % GRID_HEIGHT
        col = GRID_HEIGHT-1 - (idx // GRID_HEIGHT)
        if target_density[idx] == 1:
            pygame.draw.rect(screen, YELLOW, (row*CELL_SIZE, col*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        if init_density[idx] == 1:
            pygame.draw.rect(screen, GREEN, (row*CELL_SIZE, col*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw grid
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))

    # Draw trajectory
    for curve in curves:
        if len(curve) > 2:
            pygame.draw.lines(screen, LINE_COLOR, False, curve, LINE_WIDTH)

    # Draw active panel
    for panel in lst_panels:
        if panel != active_panel:
            panel.hide()
    active_panel.update(pygame.time.Clock().tick(60))
    active_panel.show()

    # Update the UI manager
    ui_manager.update(pygame.time.Clock().tick(60))
   # Draw the UI
    ui_manager.draw_ui(screen)

    # Update the screen
    pygame.display.flip()

# Quit Pygame
pygame.quit()
