import os, shutil
import numpy as np
import mlflow.artifacts
import pandas as pd
import yaml
from dotenv import load_dotenv
from flask import Flask, request, flash
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename

from src import seed_everything, Config
from src.features.window_generator import WindowGenerator
from src.models.da_rnn_model import DualAttentionRNN

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './upload_file'


class Model:
    def __init__(self, model_name, model_stage):
        self.weight_path = mlflow.artifacts.download_artifacts(f"models:/{model_name}/{model_stage}")
        params_path = os.path.join(self.weight_path, "params.yaml")
        self.params = yaml.safe_load(open(params_path))["train"]
        conf = Config()
        conf.window_size = self.params["window_size"]
        conf.batch_size = self.params["batch_size"]
        conf.n_future = self.params["n_future"]
        self.model = DualAttentionRNN(
            decoder_num_hidden=self.params["num_hidden_state"],
            encoder_num_hidden=self.params["num_hidden_state"],
            conf=conf,
        )

    def predict(self, data):
        conf = Config(data)
        conf.window_size = self.params["window_size"]
        conf.batch_size = self.params["batch_size"]
        conf.n_future = self.params["n_future"]
        shutil.rmtree(self.weight_path)
        w_one_target = WindowGenerator(
            data, mean_flg=True, scaler=MinMaxScaler(), conf=conf
        )
        X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()
        y_pred = self.model.predict_interval(np.expand_dims(X_test[-1], axis=0), w_one_target.conf.n_future)
        res = w_one_target.scaler.inverse_transform(y_pred[0])
        return res[:, -1].tolist()


model = Model("da_model", "Staging")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "You don't upload file", 400

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
    if file.filename.endswith(".csv"):
        data_filename = secure_filename(file.filename)
        data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
        file.save(data_path)
        data = pd.read_csv(data_path)
        os.remove(data_path)
        assert "Дата" in data.columns, "date column missing"
        data = data.set_index("Дата")
        result = model.predict(data)

    return {"value": result}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8088')
