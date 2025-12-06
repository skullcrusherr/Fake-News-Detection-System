from django import template

register = template.Library()

@register.filter(name="add_class")
def add_class(field, css):
    """
    Safely add a CSS class to a form field widget.
    If 'field' is already a rendered string, just return it.
    """
    try:
        return field.as_widget(attrs={"class": css})
    except AttributeError:
        # field is probably already a SafeString (HTML), just return as-is
        return field
