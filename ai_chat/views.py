from django.shortcuts import render
from django.http import JsonResponse

from rest_framework.decorators import api_view

from ai_chat.ai import NeuralNet, bag_of_words, tokenize, chat_func
from ai_chat.forms import UserInputForm


@api_view(["POST", "GET"])
def chat(request):
    sentence = request.data.get('sentence', '')
    if request.method == "POST":
        data = chat_func(sentence)
        return JsonResponse(data)

    return JsonResponse({"message": "GET"})


def index(request):
    form = UserInputForm()
    if request.method == "POST":
        form = UserInputForm(request.POST)
        sentence = request.POST.get('user_input', '')
        if form.is_valid():
            data = chat_func(sentence)
            context = {
                "form": form,
                "message": data["message"]
            }
            return render(request, 'index.html', context)

    if request.method == "GET":

        context = {
            "message": "",
            "form": form
        }
        return render(request, 'index.html', context=context)
    return render(request, 'index.html', context=context)
