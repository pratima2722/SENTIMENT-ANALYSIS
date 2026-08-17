import os
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# File paths for pickled model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def load_assets():
    """Loads the model and vectorizer pickles if available."""
    model, vectorizer = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f_model:
            model = pickle.load(f_model)
        with open(VECTORIZER_PATH, "rb") as f_vec:
            vectorizer = pickle.load(f_vec)
    return model, vectorizer


model, vectorizer = load_assets()

# HTML & CSS template (Embedded styled UI using Tailwind CSS)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 shadow-2xl p-6 sm:p-8">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="inline-flex items-center justify-center w-14 h-14 rounded-full bg-indigo-500/10 text-indigo-400 mb-4 border border-indigo-500/20">
                <i class="fa-solid me-0 fa-brain text-2xl"></i>
            </div>
            <h1 class="text-3xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Sentiment Analysis AI
            </h1>
            <p class="text-slate-400 text-sm mt-2">Enter text or customer feedback below to evaluate its emotional tone in real-time.</p>
        </div>

        <!-- Warning if files missing -->
        {% if not model_loaded %}
        <div class="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-300 text-sm flex items-center gap-3">
            <i class="fa-solid fa-triangle-exclamation text-amber-400 text-lg"></i>
            <div>
                <strong>Pickle files missing!</strong> Ensure <code>model.pkl</code> and <code>vectorizer.pkl</code> exist in the project root directory.
            </div>
        </div>
        {% endif %}

        <!-- Input Form -->
        <form action="/" method="POST" class="space-y-5">
            <div>
                <label for="text_input" class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                    Text Input
                </label>
                <textarea 
                    id="text_input" 
                    name="text_input" 
                    rows="4" 
                    required
                    placeholder="e.g., 'The customer service was absolutely incredible and solved my problem immediately!'"
                    class="w-full bg-slate-950/60 border border-slate-800 rounded-xl p-4 text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all resize-none text-sm"
                >{{ text_input }}</textarea>
            </div>

            <button 
                type="submit" 
                {% if not model_loaded %}disabled{% endif %}
                class="w-full py-3.5 px-6 rounded-xl font-semibold text-sm bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 active:scale-[0.99] transition-all duration-200 shadow-lg shadow-indigo-500/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                <i class="fa-solid fa-wand-magic-sparkles"></i>
                Analyze Sentiment
            </button>
        </form>

        <!-- Prediction Result Display -->
        {% if prediction %}
        <div class="mt-8 pt-6 border-t border-slate-800/80">
            <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">Analysis Output</h2>
            
            <div class="p-5 rounded-xl border flex items-center justify-between
                {% if sentiment_type == 'Positive' %}
                    bg-emerald-500/10 border-emerald-500/30 text-emerald-300
                {% elif sentiment_type == 'Negative' %}
                    bg-rose-500/10 border-rose-500/30 text-rose-300
                {% else %}
                    bg-blue-500/10 border-blue-500/30 text-blue-300
                {% endif %}">
                
                <div class="flex items-center gap-4">
                    <div class="w-12 h-12 rounded-lg flex items-center justify-center text-xl
                        {% if sentiment_type == 'Positive' %} bg-emerald-500/20 text-emerald-400
                        {% elif sentiment_type == 'Negative' %} bg-rose-500/20 text-rose-400
                        {% else %} bg-blue-500/20 text-blue-400 {% endif %}">
                        {% if sentiment_type == 'Positive' %}<i class="fa-solid fa-face-smile"></i>
                        {% elif sentiment_type == 'Negative' %}<i class="fa-solid fa-face-frown"></i>
                        {% else %}<i class="fa-solid fa-face-meh"></i>{% endif %}
                    </div>
                    <div>
                        <div class="text-xs uppercase tracking-wider font-semibold opacity-75">Predicted Sentiment</div>
                        <div class="text-xl font-bold capitalize">{{ prediction }}</div>
                    </div>
                </div>

                {% if confidence %}
                <div class="text-right">
                    <div class="text-xs uppercase tracking-wider font-semibold opacity-75">Confidence</div>
                    <div class="text-lg font-bold">{{ confidence }}%</div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    sentiment_type = None
    confidence = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()

        if text_input and model and vectorizer:
            # Vectorize input text
            transformed_text = vectorizer.transform([text_input])

            # Predict sentiment label
            raw_pred = model.predict(transformed_text)[0]
            prediction = str(raw_pred)

            # Categorize sentiment classification type for UI coloring
            pred_lower = prediction.lower()
            if any(p in pred_lower for p in ["pos", "1", "happy", "good"]):
                sentiment_type = "Positive"
            elif any(p in pred_lower for p in ["neg", "0", "bad", "sad"]):
                sentiment_type = "Negative"
            else:
                sentiment_type = "Neutral"

            # Compute prediction confidence score if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_text)[0]
                confidence = round(float(max(probs)) * 100, 1)

    return render_template_string(
        HTML_TEMPLATE,
        model_loaded=(model is not None and vectorizer is not None),
        text_input=text_input,
        prediction=prediction,
        sentiment_type=sentiment_type,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
