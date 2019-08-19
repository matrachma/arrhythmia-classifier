from utils import evaluate_model, load_saved_model
from werkzeug.utils import secure_filename
import flask
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = flask.Flask(__name__, static_url_path='/static')


def init_model():

    model_dict = {
        "sgd": load_saved_model("sgd", os.path.join(ROOT_DIR, "saved_models/sgd.h5")),
        "adam": load_saved_model("adam", os.path.join(ROOT_DIR, "saved_models/adam.h5")),
        "adagrad": load_saved_model("adagrad", os.path.join(ROOT_DIR, "saved_models/adagrad.h5")),
        "adabound": load_saved_model("adabound", os.path.join(ROOT_DIR, "saved_models/adabound.h5")),
        "amsbound": load_saved_model("amsbound", os.path.join(ROOT_DIR, "saved_models/amsbound.h5")),
        "adadelta": load_saved_model("adadelta", os.path.join(ROOT_DIR, "saved_models/adadelta.h5"))
    }

    print("Model loaded")

    return model_dict


models = init_model()


@app.route("/")
def index():
    return flask.render_template("index.html")


@app.route('/temp/images/<path:filename>')
def legacy_images(filename):
    return flask.send_from_directory("temp/images", filename)


@app.route("/evaluate", methods=["POST", "GET"])
def predict():
    file_is_ok = True
    model_selected = True
    model_select = None
    a_file_target = None

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("atr"):
            # read .atr file
            atr = flask.request.files['atr']
            a_file_name = secure_filename(atr.filename)
            # Build target
            extension = a_file_name.split(".")[1]
            if extension == "atr":
                a_file_target = os.path.join('temp/uploaded/', a_file_name)
                # Save file
                atr.save(a_file_target)
            else:
                file_is_ok = False
        else:
            file_is_ok = False
        if flask.request.files.get("hea"):
            # read .hex file
            hea = flask.request.files["hea"]
            a_file_name = secure_filename(hea.filename)
            extension = a_file_name.split(".")[1]
            if extension == "hea":
                a_file_target = os.path.join('temp/uploaded/', a_file_name)
                # Save file
                hea.save(a_file_target)
            else:
                file_is_ok = False
        else:
            file_is_ok = False
        if flask.request.files.get("dat"):
            # read .dat file
            dat = flask.request.files["dat"]
            a_file_name = secure_filename(dat.filename)
            extension = a_file_name.split(".")[1]
            if extension == "dat":
                a_file_target = os.path.join('temp/uploaded/', a_file_name)
                # Save file
                dat.save(a_file_target)
            else:
                file_is_ok = False
        else:
            file_is_ok = False

        if flask.request.form.get("model_select"):
            model_select = flask.request.form.get("model_select")
        else:
            model_selected = False

        if file_is_ok and model_selected:
            patient = str(a_file_target.split("/")[-1])
            patient = patient.split(".")[0]
            result = evaluate_model(models, patient, model_select)
        else:
            return "file not uploaded"

        return flask.render_template("result.html", data=result["data"], cm=result["confusion_matrix"],
                                     cr=result["classification_report"], classified=result["total_classified"],
                                     miss=result["miss_classified"], patient=patient)
    else:
        return flask.render_template("index.html")


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host="0.0.0.0", debug=True)
