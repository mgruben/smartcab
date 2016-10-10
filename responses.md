_**QUESTION**: Observe what you see with the agent's behavior as it takes random actions. Does the **smartcab** eventually make it to the destination? Are there any other interesting observations to note?_  
1. **Prior** to implementing the random choice among possible actions `(None, 'forward', 'left', 'right')`, the red car stayed in place.  This makes sense, because previously the action for each iteration was always `None`.  
2. **After** implementing the random choice among possible actions, as expected, the car began to move, and yes, it eventually makes it to its destination (but usually not before the deadline expires)  
3. After listening to the lectures on Q-Learning, I admit I **expected** for the car to reach its destination "better" (e.g. more directly) on subsequent iterations, but then I noticed `# TODO: Learn policy based on state, action, reward`, and realized that learning will come later.

