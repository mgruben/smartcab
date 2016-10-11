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
        
        # Variables related to Q-learning
        self.Qtable = {} # The Q-Table
        self.weights = [0, 0, 0, 0] # stores preferences for current act
        self.Qiterations = {} # used to reduce alpha
        self.alpha = 1 # initial learning rate
        self.gamma = 0.33 # Bellman discount factor

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if
        # required

    def update(self, t):
        '''
        Takes the next "best" action as defined by the Q-Learning
        algorithm.
        
        First, the next waypoint as determined by the route planner
        is obtained.
        
        Next, the smartcab determines its state (that is, the light
        color, what are the nearby cars, and in which direction
        are those cars travelling).
        
        Next, the smartcab determines the cost or benefit of the
        potential actions it could take.  These potential actions,
        as defined in the constructor, are to remain stationary, or to
        go forward, left, or right.  For more information about the
        costs and benefits, consult environment.Environment.act().
        
        The smartcab then takes the action with the highest benefit.
        
        The smartcab then peeks ahead to the highest possible benefit
        which can be obtained from the next action it would take,
        and uses that value to update its Q-Learning model.
        
        More precisely, the Q-Learning model is a table of state-action
        pairs, combined with preferences.  The adjustement of those
        preferences over time is the result of the Q-Learning algorithm.
        
        After a sufficiently-long training period (where the smartcab
        has explored the state-action pair space), it is the hope that
        the smartcab will "know" the correct action to take in order to
        get to its destination along the best route, without taking
        invalid (harmful, dangerous, illegal) actions.
        '''
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        
        
        # Light color, and the heading of traffic, if any, from
        # (1) oncoming, (2) right, and (3) left
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state with the raw values from inputs,
        # reordered to clockwise orientation
        self.state = (self.next_waypoint, inputs['light'], 
            inputs['left'], inputs['oncoming'], inputs['right'])
        
        # Select action according to your policy
        # If a state-action pair hasn't been considered yet, award
        # it a high initial Q value, to favor exploring new options
        #
        # While "exploring new options" is lamentable for a smartcab
        # with live passengers, it is also perhaps the only way to
        # traverse the reward-space during training.
        for i, action in enumerate(self.actions):
            self.weights[i] = \
                self.Qtable.setdefault((self.state, action), 5)
        
        # The next action is the highest-Q-value action
        action = self.actions[self.weights.index(max(self.weights))]
        
        # Keep track of how many times this action was chosen for
        # this state, and reduce alpha accordingly.
        #
        # If a state-action pair has never been seen before, then
        # the smartcab should learn as much as it can from this new
        # scenario.
        #
        # If a state-action pair has already occurred frequently,
        # then the smartcab should retain comparatively more of the
        # knowledge it already possesses about that state-action pair,
        # and should not "learn" as much from encountering the same
        # scenario again.
        iterations = self.Qiterations.setdefault((self.state, action), 1)
        self.Qiterations[(self.state, action)] += 1
        self.alpha = 1.0 / iterations
        
        # Execute action and get reward
        reward = self.env.act(self, action)
        
        
        # Learn policy based on state, action, reward
        
        # Look ahead one turn to find s' (state_prime)
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        self.state_prime = (self.next_waypoint, inputs['light'], inputs['left'], inputs['oncoming'], inputs['right'])
        
        # Determine the utility of the next state.
        # 
        # While previously we were concerned with which action to take,
        # here we are concerned with the greatest possible Q-value from
        # any of the next actions we _could_ take.
        #
        # Here again, if we have not previously encountered a particular
        # state-action pair, we award a high initial Q value to favor
        # exploring new options.
        for i, action_prime in enumerate(self.actions):
            self.weights[i] = self.Qtable.setdefault((self.state_prime, action_prime), 5)
        self.maxQ_new = max(self.weights)
        
        # Update Q for the current state with the just-calculated
        # utility for the next state
        # 
        # This is the equation from the "Estimating Q from Transitions"
        # Udacity video
        self.Qtable[(self.state, action)] = \
            (1.0 - self.alpha) * self.Qtable[(self.state, action)] + \
            self.alpha * (reward + self.gamma * self.maxQ_new)
        

        print "LearningAgent.update(): " + \
        "deadline = {}, inputs = {}, ".format(deadline, inputs) + \
        " action = {}, reward = {}".format(action, reward)  # [debug]

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
