rule all:
    input:
        "data/interim/data_first_handle.csv",
        "data/interim/data_with_hdb.csv",
        "data/processed/data_search.csv",
        expand("models/saved_model/"),
        "reports/validation_metric/metric_all_window.csv",
        "reports/validation_metric/average_metric.csv",
        "reports/test_metric/metric_all_window.csv",
        "reports/test_metric/average_metric.csv",
        "reports/figures/test_predict_snake.png"

rule first_handle_data:
    input:
        "data/raw/data_yandex.csv"
    output:
        "data/interim/data_first_handle.csv"
    shell:
        "python3 -m src.data.read_data {input} {output}"

rule calc_hdb:
    input:
        "data/interim/data_first_handle.csv"
    output:
        "data/interim/data_with_hdb.csv"
    shell:
        "python3 -m src.data.calculate_hdb {input} {output}"

rule calc_freq_components:
    input:
        "data/interim/data_with_hdb.csv"
    output:
        "data/processed/data_search.csv"
    shell:
        "python3 -m src.features.get_freq_components {input} {output}"

rule train_model:
    input:
        input_path = "data/processed/data_search.csv",
    params:
        window_size = "90",
        n_future = "20",
        max_epochs = "1",
    output:
        output_model_path = directory("models/saved_model/"),
        output_metric_path = ["reports/validation_metric/metric_all_window.csv",
                              "reports/validation_metric/average_metric.csv"],
        output_figure_path = "reports/figures/validation_predict_snake.png",
    shell:
        "python3 -m src.models.train_model {input} {params} {output}"

rule predict_model:
    input:
        input_path_data = "data/processed/data_search.csv",
        model_feature_path = expand("models/saved_model/"),
    params:
        window_size = "90",
        n_future = "20",
    output:
        output_metric_path = ["reports/test_metric/metric_all_window.csv",
                              "reports/test_metric/average_metric.csv"],
        output_figure_path = "reports/figures/test_predict_snake.png"
    shell:
        "python3 -m src.models.predict_model {input} {params} {output}"

rule plot_hdb:
    input:
        "data/processed/data_search.csv"
    output:
        "reports/figures/hdb.png"
    shell:
        "python3 -m src.visualization.plot_hdb {input} {output}"
