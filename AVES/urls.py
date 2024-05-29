from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing_page, name = 'landing_page'),
    path('quiz/', views.quiz, name = 'quiz'),
    path('result/', views.result, name = 'result'),
]