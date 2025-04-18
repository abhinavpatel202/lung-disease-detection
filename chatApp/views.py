from django.views.generic import TemplateView
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse 
from chat.forms import *
from neural_network.TESTING_CNN import get

import pyttsx3

class HomePage(TemplateView):
	template_name = 'index.html'

def info(request):
		return render(request, 'info.html')


def DETECTION_PAGE(request):   
    if request.method == 'POST': 
        form = INPUT_IMAGE_FORM(request.POST, request.FILES) 
  
        if form.is_valid(): 
            form.save() 
            return redirect('display_result') 
    else: 
        form = INPUT_IMAGE_FORM() 
    return render(request, 'detection.html', {'form' : form}) 
  

def predicted_results(request):
		if request.method == 'GET':
			sym1 =request.GET['filename']
			print(sym1)
		return render(request, 'detect_result.html')

def display_result(request):   
        IMAGES = INPUT_IMAGES.objects.all()
        bt_result= get('./' + str(IMAGES[len(IMAGES)-1].Input_image ))
        if request.method == 'GET':
            return render(request, 'display_result.html', {'image':IMAGES[len(IMAGES)-1].Input_image ,'result':bt_result})
        else:
            print("get audio")
            if bt_result == "Healthy Lungs":
                engine = pyttsx3.init()
                engine.say("Well, your lungs are healthy")
                engine.say("Always eat apple. Due to the presence of the antioxidant quercetin, apples have been proven to reduce lung decline and even reduce lung damage caused by smoking.")
                engine.say("A healthy body is maintained by good nutrition, regular exercise, avoiding harmful habits, making informed and responsible decisions about health, and seeking medical assistance when necessary.")
                engine.runAndWait()
            elif bt_result == "Viral Pneumonia detected":
                engine = pyttsx3.init()
                engine.say("You have detected Viral Pneumonia")
                engine.say("Most cases of viral pneumonia are mild and get better without treatment within 1 to 3 weeks. Some cases are more serious and require a hospital stay.")
                engine.say("Viral pneumonia usually isn't treated with medication and can go away on its own.")
                engine.runAndWait()
            else:
                engine = pyttsx3.init()
                engine.say("You have detected bacterial Pneumonia")
                engine.say("These medicines are used to treat bacterial pneumonia. It may take time to identify the type of bacteria causing your pneumonia and to choose the best antibiotic to treat it. If your symptoms don't improve, your doctor may recommend a different antibiotic.")
                engine.runAndWait()
            return render(request, 'display_result.html', {'image':IMAGES[len(IMAGES)-1].Input_image ,'result':bt_result})




