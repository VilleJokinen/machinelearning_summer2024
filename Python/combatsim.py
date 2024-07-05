import random
import csv


class Player:
    def __init__(self, cards, colors):
        self.cards = cards
        self.colors = colors


# One-hot encoding dictionary for colors
color_encoding = {
    'green': (1, 0, 0),
    'red': (0, 1, 0),
    'blue': (0, 0, 1)
}


def create_cards():
    cards = []
    colors = []

    card1 = random.randint(1, 8)
    card2 = random.randint(1, 9 - card1)

    # Generate card 1
    cards.append(card1)
    color1 = random.choice(list(color_encoding.values()))
    colors.append(color1)

    # Generate card 2
    cards.append(card2)
    color2 = random.choice(list(color_encoding.values()))
    colors.append(color2)

    # Generate card 3
    card3 = 10 - card1 - card2
    cards.append(card3)
    color3 = random.choice(list(color_encoding.values()))
    colors.append(color3)

    # Small chance that the sum of the cards is higher than 10 to train the model for false inputs
    if random.randint(1, 10) == 1:
        cards = []
        card_amount = 0
        while card_amount < 3:
            cards.append(random.randint(1, 10))
            card_amount += 1

    return cards, colors


def color_effectiveness(color1, color2):
    # Define effectiveness based on color interactions
    effectiveness = {
        (1, 0, 0): {(0, 1, 0): 1.5, (0, 0, 1): 1},
        (0, 1, 0): {(0, 0, 1): 1.5, (1, 0, 0): 1},
        (0, 0, 1): {(1, 0, 0): 1.5, (0, 1, 0): 1},
    }
    if color1 == color2:
        return 1
    return effectiveness[color1][color2]


def combat(p1, p2):
    results = []
    for i in range(3):
        effectiveness_p1 = color_effectiveness(tuple(p1.colors[i]), tuple(p2.colors[i]))
        effectiveness_p2 = color_effectiveness(tuple(p2.colors[i]), tuple(p1.colors[i]))

        p1_effective_card = p1.cards[i] * effectiveness_p1
        p2_effective_card = p2.cards[i] * effectiveness_p2

        if p1_effective_card > p2_effective_card:
            results.append(0)  # Player 1 wins this comparison
        elif p1_effective_card < p2_effective_card:
            results.append(1)  # Player 2 wins this comparison
        else:
            results.append(2)  # Draw for this comparison

    return results


def calculate_overall_result(results):
    p1_points = results.count(0)
    p2_points = results.count(1)

    if p1_points > p2_points:
        return 0  # Player 1 wins
    elif p2_points > p1_points:
        return 1  # Player 2 wins
    else:
        return 2  # Draw


def simulate():
    p1_cards, p1_colors = create_cards()
    p2_cards, p2_colors = create_cards()
    p1 = Player(p1_cards, p1_colors)
    p2 = Player(p2_cards, p2_colors)
    results = combat(p1, p2)
    overall_result = calculate_overall_result(results)
    return p1, p2, results, overall_result


def write_file(p1, p2, overall_result):
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow([
                "p1_card1", "p1_card2", "p1_card3",
                "p1_color1_1", "p1_color1_2", "p1_color1_3",
                "p1_color2_1", "p1_color2_2", "p1_color2_3",
                "p1_color3_1", "p1_color3_2", "p1_color3_3",
                "p2_card1", "p2_card2", "p2_card3",
                "p2_color1_1", "p2_color1_2", "p2_color1_3",
                "p2_color2_1", "p2_color2_2", "p2_color2_3",
                "p2_color3_1", "p2_color3_2", "p2_color3_3",
                "overall_result"
            ])

        # Flatten the color tuples
        p1_colors_flattened = [element for sublist in p1.colors for element in sublist]
        p2_colors_flattened = [element for sublist in p2.colors for element in sublist]

        writer.writerow(
            p1.cards + p1_colors_flattened + p2.cards + p2_colors_flattened + [overall_result]
        )


def generate_data(amount):
    # Generate specified amount of simulation data
    for _ in range(amount):
        p1, p2, results, overall_result = simulate()
        write_file(p1, p2, overall_result)


def get_middle(lst):
    return len(lst) // 2


def user_input():
    # Handle user input to determine simulation amount
    amount_to_generate = int(input("How many simulations would you like to generate? (Must be an integer)\n"))

    # Split the data to generate into chunks for cool aesthetics :cool_emoji:
    divisions = [i for i in range(1, amount_to_generate) if amount_to_generate % i == 0]
    generation_chunk = divisions[get_middle(divisions)] if divisions else 1

    generated_data = 0
    while generated_data < amount_to_generate:
        generate_data(generation_chunk)
        generated_data += generation_chunk
        print(f"Generated: {generated_data} out of {amount_to_generate} | {round(generated_data / amount_to_generate * 100)}%")


if __name__ == '__main__':
    user_input()
