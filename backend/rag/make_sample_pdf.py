from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os


def make_pdf(path="backend/data/knowledge.pdf"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    c = canvas.Canvas(path, pagesize=letter)
    text = c.beginText(40, 750)

    lines = [
        "Insurance Knowledge Base",
        "",
        "Q: How do I file a claim?",
        "A: Submit it online with photos and receipts.",
        "",
        "Q: What is a deductible?",
        "A: The amount you pay before insurance contributes.",
        "",
        "Q: How can I update my policy?",
        "A: Contact support or update it via the portal.",
    ]

    for line in lines:
        text.textLine(line)

    c.drawText(text)
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_pdf()
