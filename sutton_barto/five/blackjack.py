import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def build_policy(min_value):
    def policy(current_sum, dealer_card, usable_ace):
        if current_sum < min_value:
            return 'hit'

        return 'stick'
    return policy

def deal_card():
    # we are dealing with replacement, and the suit doesn't matter in blackjack. returns the card 'value', meaning the
    # number for regular cards, 10 for face cards, and 11 for an Ace, though aces can also be valued as 1.
    rv = random.randint(2,14)
    if rv > 10:
        if rv == 14:
            rv = 11
        else:
            rv = 10

    return rv

# return the 'best' value for a hand and whether it contains a usable ace
def eval_hand(cards):
    num_aces = cards.count(11)
    val = sum(cards)

    if num_aces > 0:
        if val < 22:
            return val, True

        for i in range(num_aces):
            val -= 10
            if val < 22:
                break

        return val, True

    return val, False

def generate_episode(policy):
    dealer_hand = [deal_card()]
    player_hand = [deal_card(),deal_card()]

    hand_sum, usable_ace = eval_hand(player_hand)
    state = (hand_sum, dealer_hand[0], usable_ace)
    states = []
    if hand_sum >= 12:
        states.append(state)
    while policy(*state) == 'hit':
        player_hand.append(deal_card())
        hand_sum, usable_ace = eval_hand(player_hand)
        if hand_sum > 21:
            return states, -1, player_hand, dealer_hand
        state = (hand_sum, dealer_hand[0], usable_ace)
        if hand_sum >= 12:
            states.append(state)

    dealer_sum, _ = eval_hand(dealer_hand)
    while dealer_sum < 17:
        dealer_hand.append(deal_card())
        dealer_sum, _ = eval_hand(dealer_hand)

    episode_return = 0
    if dealer_sum > 21 or hand_sum > dealer_sum:
        episode_return = 1
    elif hand_sum < dealer_sum:
        episode_return = -1
    elif dealer_sum == 21 and len(player_hand) == 2 and len(dealer_hand) > 2: #if the game is tied at 21, the player wins with a natural, otherwise it's a tie
        episode_return = 1

    return states, episode_return, player_hand, dealer_hand

def plot(V):
    fig = plt.figure()
    dealer = np.arange(1, 11, 1)
    hand = np.arange(12, 22, 1)
    Dealer, Hand = np.meshgrid(dealer, hand)

    axes = []
    axes.append( (fig.add_subplot(211, projection='3d'), True) )
    axes.append( (fig.add_subplot(212, projection='3d'), False) )

    for ax, usable_ace in axes:
        returns = np.array([V((h, d if d > 1 else 11, usable_ace)) for d, h in zip(np.ravel(Dealer), np.ravel(Hand))])
        Returns = returns.reshape(Dealer.shape)
        ax.plot_wireframe(Dealer, Hand, Returns)
        ax.set_xlabel('Dealer')
        ax.set_ylabel('Hand')
        ax.set_zlabel('Return')
        ax.set_title('Usable ace' if usable_ace else 'No usable ace')
    plt.show()

if __name__ == '__main__':
    pi = build_policy(20)
    episodes = 500000 #to match the diagrams in the book

    returns = {}
    for hand_sum in range(12,22):
        for dealer_card in range(2, 12):
            for usable_ace in [True, False]:
                returns[(hand_sum, dealer_card, usable_ace)] = []

    for i in xrange(episodes):
        e = generate_episode(pi)
        states = e[0]
        for s in states: #states never repeat in blackjack, so don't have to worry about multiple visits
            G = e[1]
            returns[s].append(G)

    # build value function from estimates
    def V(s):
        state_returns = returns[s]
        if len(state_returns) > 0:
            return sum(state_returns)/float(len(state_returns))
        return 0

    plot(V)
