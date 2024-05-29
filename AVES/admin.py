from django.contrib import admin
from .models import Quiz

class QuizAdmin(admin.ModelAdmin):
    list_display = ('passage', 'question', 'choices', 'correct_answer')

# Register the Quiz model with the QuizAdmin class
admin.site.register(Quiz, QuizAdmin)
