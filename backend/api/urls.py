from django.urls import path
from .views import predict_pneumonia

urlpatterns = [
    path("predict/", predict_pneumonia, name="predict"),
]
