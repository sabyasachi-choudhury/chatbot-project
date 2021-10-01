"""CHAT-BOT AND BACKEND
Required imports"""
import pickle
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn as tfl
import tensorflow as tf
import json
from flask import Flask, render_template, request
import wikipedia as wk
import wikipediaapi
from gingerit.gingerit import GingerIt
from translatepy import Translator

"""Initializing stemmer, flask app, wikipedia API, translator and autocorrect"""
parser = GingerIt()
app = Flask(__name__)
stemmer = LancasterStemmer()
wiki_wiki = wikipediaapi.Wikipedia()
ts = Translator()

"""loading data from training"""
data = pickle.load(open("training_data", "rb"))
all_stems = data['stems']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
# ERR = 0.60834564595711925
ERR = 0.68
# ERR = 0.58946292733997105
contexts = {'1': 'none'}

"""Loading JSON dataset"""
with open("intents.json") as file:
    intents = json.load(file)

"""loading trained model"""
tf.compat.v1.reset_default_graph()
net = tfl.input_data(shape=[None, len(train_x[0])])
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, 8)
net = tfl.fully_connected(net, len(train_y[0]), activation='softmax')
net = tfl.regression(net)

model = tfl.DNN(net, tensorboard_dir='tflearn_logs')
model.load('./chatbot.tflearn')

"""preprocessing data for loading options in form"""
pre_chat = []
"""Variable to allow json updates"""
update_json = False

pattern_string = []
for d in intents["intents"]:
    pattern_string.append("-".join(d["patterns"]))
pattern_string = "@".join(pattern_string)

"""variables for language and webspeech"""
lang = "en"
ttsCon = "false"


"""----------------REQUIRED FLASK BACKEND FOR WEBSITE----------------------"""
def summary(pg, sentences=3):
    summ = pg.summary.split('. ')
    summ = '. '.join(summ[:sentences])
    summ += '.'
    return summ


def clean_up(sentence):
    sentence_stems = nltk.word_tokenize(sentence)
    sentence_stems = [stemmer.stem(stem.lower()) for stem in sentence_stems]
    return sentence_stems


def bag_of_words(sentence, stems):
    sentence_stems = clean_up(sentence)
    bag = [0] * len(stems)
    for stem in sentence_stems:
        for i, w in enumerate(stems):
            if w == stem:
                bag[i] = 1
    return np.array(bag)


def classify(sentence):
    result = model.predict([bag_of_words(sentence, all_stems)])[0]
    result = [[i, c] for i, c in enumerate(result) if c > ERR]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append((classes[r[0]], r[1]))
    # print(return_list)
    return return_list


def response(sentence, user_id='1'):
    if sentence[:10].lower() == 'wikipedia:':
        s_term = sentence[10:]
        wk_res = wk.search(s_term)
        if wk_res:
            page = wiki_wiki.page(wk_res[0])
            return summary(page)
        else:
            return "Please enter a valid question"
    else:
        classification = classify(sentence)
        print(classification)
        if not classification:
            wik_suggest = response("Wikipedia:" + sentence)
            return "I'm sorry, but I couldn't understand what you meant by that." + "<br><br>However, Wikipedia says: " + wik_suggest
        if classification:
            while classification:
                for dct in intents["intents"]:
                    if dct["tag"] == classification[0][0]:
                        if 'context_set' in dct:
                            contexts[user_id] = dct['context_set']
                        if 'context_filter' not in dct or (
                                'context_filter' in dct and contexts[user_id] == dct['context_filter']):
                            return random.choice(dct['responses'])

                classification.pop(0)


def correct(sentence):
    corr = parser.parse(sentence)
    if not corr['corrections']:
        return "Autocorrect:"
    else:
        return corr["result"]


@app.route('/detect_lang_and_speech', methods=["POST"])
def detect_lang_and_speech():
    global lang, ttsCon
    lang = request.get_data().decode()
    lang = lang.split("%")
    ttsCon = lang[1].replace("25", "")
    lang = lang[0].replace("msg=", "")
    return lang


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if lang != "en":
        msg = ts.translate(msg, "en")
        msg = str(msg)
    res = response(msg)
    if ttsCon == "true" and lang != "en":
        res = ts.translate(res, lang)
        res = str(res)
    pre_chat.append("<p class='userText'>" + msg + "</p>")
    pre_chat.append("<p class='botText'>" + res + "</p>")
    return res


@app.route('/autocorrect', methods=["POST"])
def autocorrect():
    rText = request.get_data()
    rText = rText.decode()
    rText = rText[4:]
    rText = rText.replace("+", " ")
    ret = correct(rText)
    return ret


@app.route('/load_bot', methods=['POST'])
def load_bot():
    # Key:
    # #@#:small sep
    load_str = '#@#'.join(pre_chat)
    return load_str


@app.route('/get_pattern', methods=["POST"])
def get_pattern():
    set_class = request.form["setClass"]
    new_pattern = request.form["newPattern"]
    new_pattern = new_pattern.lower()
    if update_json:
        with open("intents.json") as f:
            info = json.load(f)
        for dictionary in info["intents"]:
            if dictionary["tag"] == set_class:
                dictionary["patterns"].append(new_pattern)
        with open("intents.json", "w") as f:
            json.dump(info, f, indent=2)

    return set_class + ".#." + new_pattern


@app.route("/new_class_suggest", methods=["POST"])
def new_class_suggest():
    new_class = request.form["newClass"]
    new_pattern = request.form["newPattern"]
    new_response = request.form["newResponse"]
    if update_json and new_class != "" and new_pattern != "":
        with open("intents.json") as f:
            info = json.load(f)
        info["intents"].append({
            "tag": new_class,
            "patterns": [new_pattern],
            "responses": [new_response]
        })
        with open("intents.json", "w") as f:
            json.dump(info, f, indent=2)
    return new_class + ".#." + new_pattern + ".#." + new_response


@app.route('/')
def home():
    return render_template("home_page.html")


@app.route('/principal')
def principal():
    return render_template("principal.html")


@app.route('/president')
def president():
    return render_template("president.html")


@app.route('/timetables')
def time_table():
    return render_template("school_times.html")


@app.route('/management')
def management():
    return render_template("management.html")


@app.route('/founder')
def founder():
    return render_template("founder.html")


@app.route('/registration')
def registration():
    return render_template("registration.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/fees')
def fees():
    return render_template("fees.html")


@app.route('/videos')
def videos():
    return render_template("videos.html")


@app.route('/syllabus')
def syllabus():
    return render_template("syllabus.html")


@app.route('/aims')
def aims():
    return render_template("aims.html")


@app.route('/mission')
def mission():
    return render_template("mission.html")


@app.route('/photos')
def photos():
    return render_template("gallery.html")


@app.route('/roundsquare')
def roundsquare():
    return render_template("roundsquare.html")


@app.route('/suggests')
def suggests():
    return render_template("suggests.html")


@app.route('/circulars')
def circular():
    return render_template("circulars.html")


@app.route('/exam_results')
def exam_results():
    return render_template("exam_results.html")


@app.route('/fill_options', methods=["POST"])
def fill_ops():
    cs = ".".join(classes)
    final_string = cs + "*" + pattern_string
    return final_string


"""running the website"""
if __name__ == "__main__":
    app.run()