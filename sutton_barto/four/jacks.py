import logging
import math
import collections

# Jack's car rental. Jack can choose how to divvy up his rental cars between 2 locations each night in preparation for
# the next day's rentals. See book (pg 93) for more details. This is the unmodified version.


poisson_cache = {}
def poisson_prob(n, lam):
    rv = poisson_cache.get((n, lam))
    if rv is None:
        rv = math.exp(-lam) * (lam ** n) / float(math.factorial(n))
        poisson_cache[(n, lam)] = rv
    return rv


def loc_rental_probability(start, end, rent_lam, return_lam):
    """
    calculate the rental probabilities and rewards given a particular start and end count
    :param start: the day start count
    :param end: the day end count
    :param rent_lam: the rental poisson lambda
    :param return_lam: the return poisson lambda
    :return: a list of probability/reward tuples enumerating the possible outcomes.
    """
    rv = []
    for rented in range(0, start+1):
        remaining = start - rented
        returned = end - remaining
        if returned >= 0: # feasible
            if remaining == 0:
                # cars that can be rented is capped at the number on the lot, but in theory an infinite number of customers could arrive...
                prob = 1
                for i in range(0, rented):
                    prob -= poisson_prob(i, rent_lam)
            else:
                prob = poisson_prob(rented, rent_lam)
            if end == 20:
                # lot size is capped at 20, but in theory an infinite number of cars could be returned...
                return_prob = 1
                for i in range(0, returned):
                    return_prob -= poisson_prob(i, return_lam)
            else:
                return_prob = poisson_prob(returned, return_lam)

            prob *= return_prob
            reward = 10 * rented
            rv.append((prob, reward))

    return rv


def rental_probabilities(start_state):
    """
    calculate the day activity (rental/return activity) probabilities
    :param start_state: the start of day state
    :return: a next_state -> (transition probability, *expected* reward) tuples.
    """
    rv = {}
    for l1 in range(0,21):
        for l2 in range(0,21):
            l1_probs = loc_rental_probability(start_state[0], l1, 3, 3)
            l2_probs = loc_rental_probability(start_state[1], l2, 4, 2)
            prob = 0
            reward = 0
            for l1p, l1r in l1_probs:
                for l2p, l2r in l2_probs:
                    p = l1p*l2p
                    prob += p
                    reward += p * (l1r+l2r)
            rv[(l1,l2)] = (prob, reward)

    return rv


def calculate_value(start_state, action, v, gamma, modified):
    # state is num cars at each location at the end of the day

    # do the overnight move at a cost of $2 per car, updating the counts and return
    day_start_state = (min(20, start_state[0] - action), min(20, start_state[1] + action))
    ret = -2 * abs(action)

    if modified:
        if action > 0:
            ret += 2 # can shuttle one car from 1 -> 2 for free!

        for s in day_start_state:
            if s > 10:
                ret -= 4 # $4 storage fee for > 10 cars

    # calculate the day activity probabilities
    prob_rewards = rental_probabilities(day_start_state)

    # update the return based on the probabilities, rewards and returns
    for next_state, prob_and_exp_reward in prob_rewards.iteritems():
        ret += prob_and_exp_reward[1] + prob_and_exp_reward[0] * gamma * v[next_state]

    return ret


# the state is a tuple of cars at each location, the action is the number of cars moved from location 1 to location 2.
# note that no more than 20 cars can be at either location. at most 5 cars can be moved. Policies are deterministic.
def eval_policy(v, p, modified, gamma=0.9, eps=0.1):
    iteration = 0
    while True:
        delta = 0
        v_old = collections.OrderedDict(v)
        for state, value in v_old.iteritems():
            v[state] = calculate_value(state, p[state], v_old, gamma, modified)
            delta = max(delta, abs(v[state] - value))

        iteration += 1
        if delta <= eps:
            logging.info('converged after %d iterations', iteration)
            break
        else:
            logging.info('%d: %f', iteration, delta)

        if iteration == 10000:
            raise ValueError('value function non-convergence')


def action_space(state):
    # action value is cars from 1 -> 2. negative means moving cars to location 1.
    # at most 5 cars can move, at most number of cars in other location
    minval = -1 * min(state[1], 5)
    maxval = min(state[0], 5)
    # logging.info('%s: exploring states %d -> %d', state, minval, maxval)
    return range(minval, maxval+1)


def improve_policy(v, p, modified, gamma=0.9):
    stable = True
    for state in v.iterkeys():
        temp = p[state]

        best_action = p[state]
        best_return = v[state]
        for action in action_space(state):
            ret = calculate_value(state, action, v, gamma, modified)

            if ret > best_return:
                best_action = action
                best_return = ret

        if best_action != temp:
            logging.info('updating p[%s]: %d:%f -> %d:%f (%f)', state, temp, v[state], best_action, best_return, best_return - v[state])
            p[state] = best_action
            stable = False

    return stable

def print_value_function(v):
    logging.info('    0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20')
    logging.info('    -------------------------------------------------------------------------------------------------------------------------')
    for i in range(20, -1, -1):
        values = ''
        for j in range(0, 21):
            values += str(round(v[(i, j)], 1)).rjust(5) + ' '
        logging.info('%s| %s', str(i).ljust(2), values)

if __name__ == '__main__':
    logging.basicConfig(level='INFO', format='%(message)s')

    # the policy function -- start by doing nothing
    p = collections.OrderedDict()
    for i in range(0, 21):
        for j in range(0, 21):
            p[(i,j)] = 0

    # the value function -- initialize all values to zero
    v = collections.OrderedDict()
    for i in range(0,21):
        for j in range(0,21):
            v[(i,j)] = 0

    use_modified_reward = True # True to run policy iteration with modifications from problem (1 free car shipment + $4 storage cost)
    policy_stable = False
    iteration = 0
    while not policy_stable:
        logging.info('POLICY')
        logging.info('    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20')
        logging.info('    -------------------------------------------------------------')
        for i in range(20, -1, -1):
            policy = ''
            for j in range(0, 21):
                policy += str(p[(i, j)]).ljust(2) + ' '
            logging.info('%s| %s', str(i).ljust(2), policy)

        eval_policy(v, p, use_modified_reward)
        policy_stable = improve_policy(v, p, use_modified_reward)

        iteration += 1
        logging.info('policy improvement iteration %d, stable=%s', iteration, policy_stable)
