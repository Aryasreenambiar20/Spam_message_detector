from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('svm_spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def predict_spam(text):
    text_tfidf = tfidf.transform([text])
    pred = model.predict(text_tfidf)[0]
    return 'Spam' if pred == 1 else 'Not Spam'

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    if request.method == 'POST':
        message = request.form['message']
        result = predict_spam(message)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
