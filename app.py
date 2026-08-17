import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load vectorizer and sentiment model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment.pkl", "rb") as f:
    model = pickle.load(f)

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .card-custom {
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 2.5rem;
            width: 100%;
            max-width: 550px;
        }
        .form-control {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid #334155;
            color: #f8fafc;
            border-radius: 12px;
            padding: 1rem;
            resize: none;
        }
        .form-control:focus {
            background: rgba(15, 23, 42, 0.8);
            color: #ffffff;
            border-color: #3b82f6;
            box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25);
        }
        .btn-custom {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.85rem;
            transition: all 0.3s ease;
            color: #ffffff;
        }
        .btn-custom:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
            color: #ffffff;
        }
        .result-box {
            margin-top: 1.5rem;
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 700;
            font-size: 1.25rem;
            letter-spacing: 0.5px;
        }
        .positive {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid #22c55e;
        }
        .negative {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid #ef4444;
        }
        .neutral {
            background: rgba(234, 179, 8, 0.15);
            color: #facc15;
            border: 1px solid #eab308;
        }
    </style>
</head>
<body>
    <div class="card-custom">
        <h2 class="text-center mb-1 font-weight-bold">Sentiment Analyzer</h2>
        <p class="text-center text-secondary mb-4">Enter text below to evaluate sentiment</p>
        
        <form method="POST">
            <div class="mb-3">
                <textarea class="form-control" name="text" rows="4" placeholder="e.g., The product quality is exceptional and fast delivery!" required>{{ user_text }}</textarea>
            </div>
            <button type="submit" class="btn btn-custom w-100">Analyze Sentiment</button>
        </form>

        {% if prediction %}
        <div class="result-box {{ sentiment_class }}">
            Result: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    sentiment_class = ""
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "")
        if user_text:
            text_vec = vectorizer.transform([user_text])
            result = model.predict(text_vec)[0]

            val = str(result).lower()
            if val in ["1", "positive", "pos"]:
                prediction = "Positive 😊"
                sentiment_class = "positive"
            elif val in ["0", "negative", "neg"]:
                prediction = "Negative 😞"
                sentiment_class = "negative"
            else:
                prediction = f"Output: {result}"
                sentiment_class = "neutral"

    return render_template_string(
        HTML_LAYOUT,
        prediction=prediction,
        sentiment_class=sentiment_class,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
