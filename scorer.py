from io import BytesIO
import pickle

import numpy as np

from flask import Flask, request, make_response

app = Flask(__name__)

models = pickle.load(open('models.pickle', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    result = 'Please fill in and submit the form above.'
    if 'dialogue' in request.form:
        text = request.form['dialogue']
        tfidf_1 = models['tfidf_1'].transform([text])
        tfidf_2 = models['tfidf_2'].transform([text])
        
        lr_predict_1 = models['lr_1'].predict(tfidf_1)
        lr_predict_proba_1 = models['lr_1'].predict_proba(tfidf_1)
        
        lr_predict_2 = models['lr_2'].predict(tfidf_2)
        lr_predict_proba_2 = models['lr_2'].predict_proba(tfidf_2)
        
        svm_predict_1 = models['svm_1'].predict(tfidf_1)
        svm_predict_proba_1 = models['svm_1'].decision_function(tfidf_1)
        
        svm_predict_2 = models['svm_2'].predict(tfidf_2)
        svm_predict_proba_2 = models['svm_2'].decision_function(tfidf_2)
        
        svm_predict_proba_1_norm = (svm_predict_proba_1 - svm_predict_proba_1.min(axis=1)[:, np.newaxis]) / (
            (svm_predict_proba_1.max(axis=1) - svm_predict_proba_1.min(axis=1)))[:, np.newaxis]

        svm_predict_proba_2_norm = (svm_predict_proba_2 - svm_predict_proba_2.min(axis=1)[:, np.newaxis]) / (
            (svm_predict_proba_2.max(axis=1) - svm_predict_proba_2.min(axis=1)))[:, np.newaxis]
        
        f_all = np.hstack((lr_predict_1, lr_predict_proba_1,
                           lr_predict_2, lr_predict_proba_2,
                           svm_predict_1, svm_predict_proba_1_norm,
                           svm_predict_2, svm_predict_proba_2_norm))
        
        lgbm_predict_proba = models['lgbm'].predict_proba(f_all)
        lgbm_preds = list(map(lambda row: np.array([1 if proba >= max(0.37, models['scores'][k])
                                            else 0 for k, proba in enumerate(row)]), lgbm_predict_proba))
        result = list(models['mlb'].inverse_transform(np.vstack(lgbm_preds))[0])
        if result == []: result = ['drama']
        result = ', '.join(result)
    return f"""
<!DOCTYPE html>
<title>Assign genres</title>
<form method="post" action="/" enctype="multipart/form-data">
    <label for="dialogue">Text to score</label><br>
    <textarea rows="25" cols="90" name="dialogue" id="dialogue" required></textarea><br>
    <br>
    <input type="submit" value="Send">
</form>
<hr>
{result}
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
