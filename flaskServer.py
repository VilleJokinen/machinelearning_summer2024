from flask import Flask, request, jsonify
import torch
from combatsimModel import CardCombatModel

app = Flask(__name__)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CardCombatModel().to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    p1_cards = torch.tensor(data['p1_cards']).unsqueeze(0).float().to(device)
    p2_cards = torch.tensor(data['p2_cards']).unsqueeze(0).float().to(device)
    with torch.no_grad():
        outputs = model(p1_cards, p2_cards)
        _, predicted = torch.max(outputs.data, 1)
        result = predicted.item() + 1  # Adjust result to be in range [1, 3]
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
