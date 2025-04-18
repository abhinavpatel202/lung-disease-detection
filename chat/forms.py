from django import forms
from .models import INPUT_IMAGES


class INPUT_IMAGE_FORM(forms.ModelForm):   
    class Meta: 
        model = INPUT_IMAGES
        fields = ['Input_image'] 
