from django.db import models
from django.contrib.auth import get_user_model
from django.utils.timezone import now
from django.urls import reverse


# Create your models here.
User = get_user_model()

class INPUT_IMAGES(models.Model): 
    Input_image = models.ImageField(upload_to='static/images/') 
