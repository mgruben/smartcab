_**QUESTION**: Observe what you see with the agent's behavior as it takes random actions. Does the **smartcab** eventually make it to the destination? Are there any other interesting observations to note?_  
1. **Prior** to implementing the random choice among possible actions `(None, 'forward', 'left', 'right')`, the red car stayed in place.  This makes sense, because previously the action for each iteration was always `None`.  
2. **After** implementing the random choice among possible actions, as expected, the car began to move, and yes, it eventually makes it to its destination (but usually not before the deadline expires)  
3. After listening to the lectures on Q-Learning, I admit I **expected** for the car to reach its destination "better" (e.g. more directly) on subsequent iterations, but then I noticed `# TODO: Learn policy based on state, action, reward`, and realized that learning will come later.  
4. It's a little hard to tell, but it also appears like the agent is, at this point, doing wrong things (e.g. disobeying traffic laws in ways that are likely to bring harm to itself and others).  From what I could tell in `environment.py`, such harmful actions are classified as `Invalid Moves`, and result in a `reward` of `-1.0`.  There were definitely a few `rewards` of `-1.0` in the simulation.  
5. So, at this point, it seems like we've successfully given movement to an actor which is likely to (unintentionally) inflict harm.  Yikes!

_**QUESTION**: What states have you identified that are appropriate for modeling the **smartcab** and environment? Why do you believe each of these states to be appropriate for this problem?_  
1. Every state in `inputs` (that is, in `self.env.sense(self)`) is important for modeling the **smartcab** and its environment.  
..1. The state of the light (e.g. `red, green`) is required to know whether the desired next action is presently allowed.  
..2. Similarly, the location and heading of other cars is important for knowing whether or not our `agent` needs to adjust its desired heading.  Without knowing this information, our `agent` will be unable to perform **collision avoidance**.

_**OPTIONAL**: How many states in total exist for the **smartcab** in this environment? Does this number seem reasonable given that the goal of Q-Learning is to learn and make informed decisions about each state? Why or why not?_  
1. From mere permutations of the `inputs` vector, there are 128 distinct states.  This is because there are two states for `light`, and four for each of `oncoming`, `right`, and `left` (specifically, `None`, `forward`, `right`, and `left`).
```
e.g.:
{'green', 'oncoming': None, 'right': None, 'left': None}
{'red', 'oncoming': None, 'right': None, 'left': 'left'}
{'red', 'oncoming': 'right', 'right': None, 'left': 'forward'}
```
2. This high number of states seems correct for the `agent` to have a full understanding of the intersection and its possible next actions.  Without an understanding this full, it seems unlikely that a **smartcab** would be able to explore alternative routes in case the chosen one is blocked.  
..* For instance, consider a smartcab approaching a green light, correctly intending to go straight through the intersection.  If another car approaches the intersection from the smartcab's right, intends to go forward, and should stop at the red light but doesn't, the smartcab must correctly ascertain that it has to stop, or else it will contribute to an accident it could otherwise have prevented, even though it's not the case that it caused the accident all by itself.  
..* Even though there are 128 distinct possible states, many of these states are invalid, since they would cause an accident even in the absence of the smartcab.  However, even strange accidents happen occasionally, so while the likelihood of many of the potential states is low, it's not zero, so the smartcab can't assume they'll never occur.
