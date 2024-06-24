import random
import csv


class Player:
    def __init__(self, cards, colors, points):
        self.cards = cards
        self.colors = colors
        self.points = points


color_mapping = {'green': 0, 'red': 1, 'blue': 2}


def create_cards():
    cards = []
    colors = []

    card1 = random.randint(1, 8)
    card2 = random.randint(1, 9 - card1)

    # card 1
    cards.append(card1)
    colors.append(random.choice([0, 1, 2]))

    # card 2
    cards.append(card2)
    colors.append(random.choice([0, 1, 2]))

    # card 3
    card3 = 10 - card1 - card2
    cards.append(card3)
    colors.append(random.choice([0, 1, 2]))

    if random.randint(1, 10) == 1:
        cards = []
        card_amount = 0
        while card_amount < 3:
            cards.append(random.randint(1, 10))
            card_amount += 1

    return cards, colors


def color_effectiveness(color1, color2):
    effectiveness = {
        0: {1: 1.5, 2: 1},
        1: {2: 1.5, 0: 1},
        2: {0: 1.5, 1: 1},
    }
    if color1 == color2:
        return 1
    return effectiveness[color1][color2]


def calculate_points(p1, p2):
    for i in range(3):
        effectiveness_p1 = color_effectiveness(p1.colors[i], p2.colors[i])
        effectiveness_p2 = color_effectiveness(p2.colors[i], p1.colors[i])

        p1_effective_card = p1.cards[i] * effectiveness_p1
        p2_effective_card = p2.cards[i] * effectiveness_p2


        if p1_effective_card > p2_effective_card:
            p1.points += 1
        elif p1_effective_card < p2_effective_card:
            p2.points += 1


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
    p1_cards, p1_colors = create_cards()
    p2_cards, p2_colors = create_cards()
    p1 = Player(p1_cards, p1_colors, 0)
    p2 = Player(p2_cards, p2_colors, 0)
    combat(p1, p2)
    result = determine_result(p1, p2)
    return p1, p2, result

def write_file(p1, p2, result):
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "p1_card1", "p1_color1", "p1_card2", "p1_color2", "p1_card3", "p1_color3",
                "p2_card1", "p2_color1", "p2_card2", "p2_color2", "p2_card3", "p2_color3", "result"
            ])

        writer.writerow(
            p1.cards + p1.colors + p2.cards + p2.colors + [result]
        )

def generate_data(amount):
    for _ in range(amount):
        p1, p2, result = simulate()
        write_file(p1, p2, result)

def get_middle(lst):
    return len(lst) // 2

def user_input():
    amount_to_generate = int(input("How many simulations would you like to generate? (Must be an integer) \n"))

    divisions = [i for i in range(1, amount_to_generate) if amount_to_generate % i == 0]
    generation_chunk = divisions[get_middle(divisions)] if divisions else 1

    generated_data = 0
    while generated_data < amount_to_generate:
        generate_data(generation_chunk)
        generated_data += generation_chunk
        print(f"Generated: {generated_data} out of {amount_to_generate} | {round(generated_data / amount_to_generate * 100)}%")

if __name__ == '__main__':
    user_input()
