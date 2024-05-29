from django.db import models

class Quiz(models.Model):
    passage = models.TextField()
    question = models.CharField(max_length=255)
    choices = models.CharField(max_length=1000)
    correct_answer = models.CharField(max_length=100)

    def __str__(self):
        return self.question