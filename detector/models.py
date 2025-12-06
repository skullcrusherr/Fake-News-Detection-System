from django.db import models
from django.conf import settings


class PredictionLog(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )
    text = models.TextField()
    label = models.CharField(max_length=20)  # REAL / FAKE / UNSURE / UNKNOWN
    real_prob = models.FloatField(null=True, blank=True)
    fake_prob = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def short_text(self, max_len=80):
        t = self.text.replace("\n", " ")
        return (t[:max_len] + "…") if len(t) > max_len else t

    def __str__(self):
        return f"{self.label} @ {self.created_at:%Y-%m-%d %H:%M}"
