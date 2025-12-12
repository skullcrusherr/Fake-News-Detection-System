from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.db.models import Count
from django.contrib.auth import logout as auth_logout

from .ml_model import predict_news
from .models import PredictionLog

from .forms import ImageUploadForm
from .ocr_utils import extract_text_from_image
from django.core.files.storage import default_storage


@login_required
def home(request):
    prediction = None
    probabilities = None
    user_text = ""

    if request.method == "POST":
        user_text = request.POST.get("news_text", "")
        if user_text.strip():
            prediction, probabilities = predict_news(user_text)

            real_p = probabilities.get("REAL") if probabilities else None
            fake_p = probabilities.get("FAKE") if probabilities else None

            PredictionLog.objects.create(
                user=request.user,
                text=user_text,
                label=prediction or "UNKNOWN",
                real_prob=real_p,
                fake_prob=fake_p,
            )

    # last 10 predictions (current user)
    recent_logs = PredictionLog.objects.filter(user=request.user).order_by("-created_at")[:10]

    # global stats for pie chart
    label_counts = PredictionLog.objects.values("label").annotate(count=Count("label"))
    counts = {"REAL": 0, "FAKE": 0, "UNSURE": 0}
    for row in label_counts:
        label = row["label"]
        if label in counts:
            counts[label] = row["count"]

    total_predictions = sum(counts.values())

    context = {
        "prediction": prediction,
        "probabilities": probabilities,
        "user_text": user_text,
        "recent_logs": recent_logs,
        "total_predictions": total_predictions,
        "real_count": counts["REAL"],
        "fake_count": counts["FAKE"],
        "unsure_count": counts["UNSURE"],
    }
    return render(request, "detector/home.html", context)


def register(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()

    return render(request, "detector/register.html", {"form": form})

def logout_view(request):
    auth_logout(request)
    return redirect("login")


@login_required
def history(request):
    logs = PredictionLog.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "detector/history.html", {"logs": logs})


User = get_user_model()


@user_passes_test(lambda u: u.is_staff)
def users_list(request):
    users = User.objects.all().order_by("username")
    return render(request, "detector/users_list.html", {"users": users})

from django.core.files.storage import default_storage
from django.conf import settings  # add this import

@login_required
def image_check(request):
    prediction = None
    probabilities = None
    extracted_text = ""
    uploaded_image_url = None
    form = ImageUploadForm()

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data["image"]

            saved_path = default_storage.save(f"uploads/{img_file.name}", img_file)
            full_path = default_storage.path(saved_path)

            # ✅ URL to show in template
            uploaded_image_url = settings.MEDIA_URL + saved_path

            extracted_text = extract_text_from_image(full_path)

            if extracted_text.strip():
                prediction, probabilities = predict_news(extracted_text)

                real_p = probabilities.get("REAL") if probabilities else None
                fake_p = probabilities.get("FAKE") if probabilities else None

                PredictionLog.objects.create(
                    user=request.user,
                    text=extracted_text,
                    label=prediction or "UNKNOWN",
                    real_prob=real_p,
                    fake_prob=fake_p,
                )

    return render(request, "detector/image_check.html", {
        "form": form,
        "prediction": prediction,
        "probabilities": probabilities,
        "extracted_text": extracted_text,
        "uploaded_image_url": uploaded_image_url,  # ✅
    })
