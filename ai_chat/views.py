import random
import json
import torch
import os
from ai_chat.train import NeuralNet, bag_of_words, tokenize

from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.conf import settings


@api_view(["POST", "GET"])
def chat(request):
    file_path = os.path.abspath(os.path.join(
        settings.BASE_DIR, 'ai_chat', 'intents.json'))
    with open(file_path, 'r') as json_data:
        intents = json.load(json_data)

    # load saved data after training step
    FILE = os.path.abspath(os.path.join(
        settings.BASE_DIR, 'ai_chat', "data.pth"))
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    # Sets the module in evaluation mode.
    model.eval()

    #

    if request.method == "POST":
        sentence = request.data.get('sentence', '')
        # sentence = tokenize(sentence)
        X = bag_of_words(tokenize(sentence), all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    data = {
                        "message": random.choice(intent['responses']),
                    }
                    # print(f"{bot_name}: {random.choice(intent['responses'])}")
                    return JsonResponse(data)
        else:
            # print(f"{bot_name}: I do not understand...")
            data = {
                "message": "I do not understand...",
            }
            return JsonResponse(data)

    return JsonResponse({"message": "GET"})
