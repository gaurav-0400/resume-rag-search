from flask import Flask, render_template, request
from query import search_resume

app= Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer=""
    question=""

    if request.method == "POST":
        question= request.form.get("question", "").strip()

        if question:
            answer = search_resume(question)

    return render_template(
        "index.html",
        question= question,
        answer= answer)

    
if __name__ == "__main__":
    
    app.run(debug=True)

