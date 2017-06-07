import logging
# policy iteration on a gridworld. numbering is 1-14, left to right, up to down. 0 and 15 (upper left square and bottom
# right square) are terminal states. each transition results in a -1 reward. transitioning out of the grid results in a
# -1 reward and returning to the current state. There is no discounting.


def step(state, action):
    """    
    :param state: the current state (1-14) 
    :param action: the next action (udlr)
    :return: the next state + reward. Note 0 is the terminal state
    """
    allowed = set(['l', 'r','u','d'])

    if state < 1 or state > 14:
        raise ValueError(state)
    if action not in allowed:
        raise ValueError(action)

    if state <= 3:
        allowed.remove('u')
    if state >= 12:
        allowed.remove('d')
    if state % 4 == 0:
        allowed.remove('l')
    if (state - 3) % 4 == 0:
        allowed.remove('r')

    if action in allowed:
        if action == 'u':
            return state - 4, -1
        if action == 'l':
            return state - 1, -1
        if action == 'r':
            return (state + 1) % 15, -1 #15 means terminal state
        if action == 'd':
            return (state + 4) % 15, -1
    else:
        return state, -1


def random_policy(action, state):
    return 0.25


if __name__ == '__main__':
    logging.basicConfig(level='INFO', format='%(message)s')
    v = {i: 0 for i in range(1,15)}
    iteration = 0
    while True:
        delta = 0
        v_old = dict(v)
        for state, state_value in v.iteritems():
            temp = v[state]
            updated = 0
            for action in ('l', 'r', 'u', 'd'):
                action_prob = random_policy(action, state)
                next_state, reward = step(state, action) # transitions are deterministic!
                next_state_value = v_old[next_state] if next_state != 0 else 0
                updated += action_prob * (reward + 1.0 * next_state_value) # no discounting!
            v[state] = updated
            delta = max(delta, abs(temp - v[state]))

        if delta < 0.001:
            logging.info('converged on iteration %d', iteration)
            break

        iteration += 1
        logging.info('state values on iteration %d', iteration)
        logging.info('\t       %.1f %.1f %.1f', v[1], v[2], v[3])
        logging.info('\t %.1f %.1f %.1f %.1f', v[4], v[5], v[6], v[7])
        logging.info('\t %.1f %.1f %.1f %.1f', v[8], v[9], v[10], v[11])
        logging.info('\t       %.1f %.1f %.1f', v[12], v[13], v[14])

        if iteration == 1000:
            logging.info('non-convergence: stopping on iteration %d', iteration)
            break