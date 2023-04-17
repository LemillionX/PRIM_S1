import pygame
import pygame_gui
import callbacksUI as ui

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Grid dimensions
CELL_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = CELL_SIZE * GRID_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_HEIGHT
WIDGET_HEIGHT = CELL_SIZE * 2
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Button settings
BUTTON_WIDTH = 100
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
save_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(5, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT),
    text='Save',
    manager=ui_manager,
    container=tool_bar_container
)

reset_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(BUTTON_WIDTH + 2*BUTTON_LEFT, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT),
    text='Reset',
    manager=ui_manager,
    container=tool_bar_container
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

# Define callbacks for buttons
def button_click():
    print("Button clicked!")

def save_trajectory():
    file_name = ui.prompt_file()
    ui.saveToJSON(visited_cells[0], GRID_HEIGHT, file_name)
    print("Trajectiry saved here : ", file_name)

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
                    drawing = True
                    curves.append([event.pos])
                    visited_cells.append([[event.pos[0]//CELL_SIZE, GRID_HEIGHT-1 - event.pos[1]//CELL_SIZE]])
        elif event.type == pygame.MOUSEMOTION and drawing:
            # If the left mouse button is pressed and moving, continue the curve
            curves[-1].append(pygame.mouse.get_pos())
            if visited_cells[-1][-1] != [pygame.mouse.get_pos()[0]//CELL_SIZE, GRID_HEIGHT-1 - pygame.mouse.get_pos()[1]//CELL_SIZE]:
                visited_cells[-1].append([pygame.mouse.get_pos()[0]//CELL_SIZE, GRID_HEIGHT-1 - pygame.mouse.get_pos()[1]//CELL_SIZE])
        elif event.type == pygame.MOUSEBUTTONUP:
            # If the left mouse button is released, add the curve to all curves
            if event.button == 1:
                drawing = False

        # Check for button press event
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == save_button:
                    save_trajectory()
                if event.ui_element == reset_button:
                    reset()

        # Update the UI manager with the event
        ui_manager.process_events(event)



    # Redraw the screen
    screen.fill(WHITE)
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WINDOW_WIDTH, y))
    for curve in curves:
        if len(curve) > 2:
            pygame.draw.lines(screen, LINE_COLOR, False, curve, LINE_WIDTH)


    # Update the UI manager
    ui_manager.update(pygame.time.Clock().tick(60))
   # Draw the UI
    ui_manager.draw_ui(screen)

    # Update the screen
    pygame.display.flip()

# Quit Pygame
pygame.quit()
