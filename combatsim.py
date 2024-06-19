import random
import csv

class Player:
    def __init__(self, cards, points):
        self.cards = cards
        self.points = points


def create_cards():
    cards = []

    card1 = random.randint(1, 8)
    card2 = random.randint(1, 9 - card1)

    # card 1:
    # card1 = random.randint(0,10)
    cards.append(card1)

    # card 2
    # card2 = random.randint(0,10-card1)
    cards.append(card2)

    # card 3
    card3 = 10 - card1 - card2
    cards.append(card3)

    if random.randint(1,10) == 1:
        cards = []
        card_amount = 0
        while card_amount < 3:
            cards.append(random.randint(1, 10))
            card_amount += 1

    return cards


def calculate_points(p1, p2):
    i = 0
    while i < 3:
        if p1.cards[i] > p2.cards[i]:
            p1.points += 1
        elif p1.cards[i] < p2.cards[i]:
            p2.points += 1
        i += 1


def combat(p1, p2):
    calculate_points(p1, p2)

    if sum(p1.cards) != 10:
        p1.points = 0
        p2.points = 3

    if sum(p2.cards) != 10:
        p2.points = 0
        p1.points = 3

    if sum(p1.cards) != 10 and sum(p2.cards) != 10:
        p1.points = 0
        p2.points = 0

    return p1.points, p2.points


def determine_result(p1, p2):
    if p1.points == p2.points:
        return 3
    if p1.points > p2.points:
        return 1
    if p1.points < p2.points:
        return 2


def simulate():
    p1 = Player(create_cards(), 0)
    p2 = Player(create_cards(), 0)
    combat(p1, p2)
    result = determine_result(p1, p2)
    return p1, p2, result


def write_file(p1, p2, result):
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["p1_card1", "p1_card2", "p1_card3", "p2_card1", "p2_card2", "p2_card3", "result"])

        writer.writerow(p1.cards + p2.cards + [result])


def generate_data(amount):
    i = 0
    while i < amount:
        p1, p2, result = simulate()
        write_file(p1, p2, result)
        i+=1


def get_middle(list):
    middleIndex = int(len(list)/2)
    return middleIndex


def user_input():
    amount_to_generate = int(input("How many simulations would you like to generate? (Must be an integer) \n"))

    divisions = []
    i = 1
    while i < amount_to_generate:
        if amount_to_generate % i == 0:
            divisions.append(i)
        i += 1

    if (len(divisions) == 0):
        generation_chunk = 1
    else:
        middleIndex = get_middle(divisions)
        generation_chunk = divisions[middleIndex]

    generated_data = 0
    while generated_data < amount_to_generate:
        generate_data(generation_chunk)
        generated_data += generation_chunk
        print(f"Generated: {generated_data} out of {amount_to_generate} | {round(generated_data / amount_to_generate * 100)}%" )


if __name__ == '__main__':
    user_input()