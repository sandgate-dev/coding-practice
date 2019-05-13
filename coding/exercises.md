# EXERCISES

## blackjack

Coding interview @ freenome \(initial version\)

```python
import os
from random import shuffle
from collections import defaultdict

cards = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10, 'A':[1, 10]}
players = defaultdict(list)


def init_deck():
    deck = list(cards.keys()) * 4
    shuffle(deck)
    return deck


def hand_cards(player):
    for i in range(2):
        players[player].append(deck_of_cards.pop())
    return players


def calc_init_hand(player):
    if player is 'Player' and players[player].count('A') > 1:
        del players[player][-1:]
        players[player].append('1')

    values = [cards.get(k)[1] if k is 'A' else cards.get(k) for k in players[player]]
    return sum(values)


def add_card(player):
    return players[player].append(deck_of_cards.pop())


def print_hand():
    print('Dealer: {} \t- {}'.format(calc_init_hand('Dealer'), players['Dealer'][1:]))
    print('Player: {} \t- {}'.format(calc_init_hand('Player'), players['Player']))


def total(dealer, player):
    if calc_init_hand(dealer) < calc_init_hand(player) <= 21 or \
            calc_init_hand(dealer) > 21 >= calc_init_hand(player):

        print_hand()
        print('Player wins\n')
        choice = input("Do you want to play [a]gain or [q]uit: ").lower()
        if choice == 'a':
            players.clear()
            game()
        else:
            exit()
    elif calc_init_hand(player) < calc_init_hand(dealer) <= 21 or \
            calc_init_hand(player) > 21 >= calc_init_hand(dealer):

        print_hand()
        print('Dealer wins\n')
        choice = input("Do you want to play [a]gain or [q]uit: ").lower()
        if choice == 'a':
            players.clear()
            game()
        else:
            exit()
    elif calc_init_hand(player) > 21 < calc_init_hand(dealer):

        print_hand()
        print('Both loose')
        exit()

    return


def game():
    global deck_of_cards

    # Initialize deck of cards
    deck_of_cards = init_deck()
    hand_cards('Dealer')
    hand_cards('Player')

    while True:
        print_hand()

        choice = input("Do you want to [H]it, [S]tand, [A]gain or [Q]uit: ").lower()
        # os.system('clear')

        if choice == "h":
            add_card('Player')
            if calc_init_hand('Dealer') < 17:
                add_card('Dealer')

            total('Dealer', 'Player')
        elif choice == "s":
            if calc_init_hand('Dealer') < 17:
                add_card('Dealer')

            total('Dealer', 'Player')
        elif choice == 'a':
            players.clear()
            game()
        elif choice == "q":
            print("Bye!")
            exit()


if __name__ == "__main__":
    game()
```

