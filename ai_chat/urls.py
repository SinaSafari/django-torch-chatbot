from django.urls import path
from ai_chat.views import chat, index

urlpatterns = [
    path('api/chat', chat, name="chat"),
    path('', index, name="index"),
]
