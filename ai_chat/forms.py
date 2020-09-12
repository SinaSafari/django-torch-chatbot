from django import forms


class UserInputForm(forms.Form):
    user_input = forms.CharField(label='Ask please', max_length=100)
