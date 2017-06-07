import logging
from collections import OrderedDict
from decimal import Decimal
# code for the "gambler's problem": Essentially a gambler places bets on a weighted coin flip coming up heads, trying
# to get to $100. He can only bet how much money he has (he loses when he gets to $0 and there is no benefit to
# winning > $100). # use value iteration to try to solve the problem.



def action_value(s, a, p_h, gamma, v):
    if a == 0:
        return v[s]

    lose_state, win_state = s - a, s + a
    if lose_state == 0:
        lose_val = 0
    else:
        lose_val = (1 - p_h) * gamma * v[lose_state]
    if win_state == 100:
        win_val = p_h  # (reward of 1)
    else:
        win_val = p_h * gamma * v[win_state]

    return win_val + lose_val


def do_value_iteration(p_h, gamma, eps):
    iterations = 0
    delta = eps + 1
    v = OrderedDict()
    for i in range(1,100):
        v[i] = Decimal(0)
    while delta > eps:
        delta = 0
        v_old = dict(v)
        iterations += 1
        for s in range(1,100):
            temp = v[s]
            actions = range(1, min(s, 100 - s)+1) #can only bet what you have or enough to reach $100
            for a in actions:
                val = action_value(s, a, p_h, gamma, v_old)
                if val > v[s]:
                    v[s] = val

            delta = max(delta, abs(temp - v[s]))

    logging.info('value function converged after %d iterations', iterations)

    return v

def build_policy(p_h, gamma, eps, v):
    p = OrderedDict()
    for s in range(1, 100):
        actions = range(1, min(s, 100 - s) + 1)
        best_action = 0
        best_val = 0
        for a in actions:
            val = action_value(s, a, p_h, gamma, v)
            if s == 40:
                logging.info('40: %d: %10f', a, val)

            #little hacky, just pick a larger bet when we are sure that it will be better
            if val > best_val+eps:
                best_val = val
                best_action = a

        p[s] = best_action
    return p

if __name__ == '__main__':
    logging.basicConfig(level='INFO', format='%(message)s')
    gamma = Decimal(1)
    eps = Decimal(1e-20)

    p_h = Decimal(4)/10 # probability of a heads coin flip


    v = do_value_iteration(p_h, gamma, eps)
    p = build_policy(p_h, gamma, eps, v)

    # plot the policy
    # import matplotlib.pyplot as plt
    # plt.step(p.keys(), p.values())

