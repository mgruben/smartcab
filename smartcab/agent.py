import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        '''
        sets self.env = env, state = None, next_waypoint = None,
        and a default color
        '''
        super(LearningAgent, self).__init__(env)
        
        # override color
        self.color = 'red'
        
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        
        # TODO: Initialize any additional variables here
        self.actions = (None, 'forward', 'left', 'right')
        self.state = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if
        # required

    def update(self, t):
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = inputs
        
        # TODO: Select action according to your policy
        action = random.choice(self.actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}," + \
            "action = {}, reward = {}".format(deadline, \
            inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # create environment (also adds some dummy traffic)
    e = Environment()
    
    # create agent
    a = e.create_agent(LearningAgent)
    
    # specify agent to track
    e.set_primary_agent(a, enforce_deadline=True)
    # NOTE: You can set enforce_deadline=False while debugging to
    # allow longer trials
    
    # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.1, display=True)
    # NOTE: To speed up simulation, reduce update_delay and/or set
    # display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit
    # Ctrl+C on the command-line


if __name__ == '__main__':
    run()
