from django.urls import path
from ai_chat.views import chat

urlpatterns = [
    path('', chat, name="chat")
]
