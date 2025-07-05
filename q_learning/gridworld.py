import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = None
        self.max_steps = 50
        self.current_step = 0
        
    def reset(self):
        self.state = (0, 0)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        self.current_step += 1
        
        # Current position
        x, y = self.state
        
        # Try to move
        if action == 0:    # up
            new_state = (max(0, x-1), y)
        elif action == 1:  # right
            new_state = (x, min(self.size-1, y+1))
        elif action == 2:  # down
            new_state = (min(self.size-1, x+1), y)
        elif action == 3:  # left
            new_state = (x, max(0, y-1))
            
        self.state = new_state
        
        # Check if goal reached
        done = (self.state == (self.size-1, self.size-1)) or (self.current_step >= self.max_steps)
        reward = 10 if self.state == (self.size-1, self.size-1) else -1
        
        return self.state, reward, done
