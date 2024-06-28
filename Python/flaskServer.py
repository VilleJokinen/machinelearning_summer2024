from flask import Flask, request, jsonify
import torch
from combatsimModel import CardCombatModel

app = Flask(__name__)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CardCombatModel().to(device)
model.load_state_dict(torch.load('combatsimModel.pth', map_location=device))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    p1_cards = torch.tensor(data['p1_cards']).float().to(device)
    p1_colors = torch.tensor(data['p1_colors']).float().to(device)
    p2_cards = torch.tensor(data['p2_cards']).float().to(device)
    p2_colors = torch.tensor(data['p2_colors']).float().to(device)

    # Concatenate all inputs into a single tensor
    inputs = torch.cat((p1_cards, p1_colors, p2_cards, p2_colors), dim=0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        result = predicted.item()

    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
