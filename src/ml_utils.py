import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_model_by_window(
    init_params,
    fit_params,
    data,
    test_init,
    test_finish,
    window_type="rolling",
    train_size=60,
    test_size=1,
    metric="rmse",
):
    """
    Utiliza una ventana móvil o recursiva para evaluar un modelo ETS en un periodo de prueba específico.

    Parámetros:
    - init_params: Parámetros iniciales para el modelo ETS.
    - fit_params: Parámetros de ajuste para el modelo ETS.
    - data: DataFrame con la serie temporal a evaluar.
    - test_init: Fecha de inicio del periodo de prueba (formato 'YYYY-MM-DD').
    - test_finish: Fecha de fin del periodo de prueba (formato 'YYYY-MM-DD').
    - window_type: Tipo de ventana a utilizar ('rolling' o 'expanding').
    - train_size: Tamaño de la ventana de entrenamiento (en meses).
    - test_size: Tamaño de la ventana de prueba (en meses).
    - metric: Métrica a utilizar para evaluar el modelo ('rmse' o 'mae').
    """

    metric_list = []
    temp_data = data.reset_index().copy()
    ind_test_init = temp_data[temp_data["mes"] == test_init].index[0]

    # Numero de meses entre test_init y test_finish
    months = round(
        (pd.to_datetime(test_finish) - pd.to_datetime(test_init)).days * (12 / 365)
    )

    # Calculamos el numero total de pasos a evaluar
    total_steps = months / test_size

    if total_steps % 1 != 0:
        raise ValueError(
            "El tamaño de la ventana no es un divisor del rango de fechas."
        )
    else:
        total_steps = int(total_steps)

    if window_type == "rolling":
        # Aplicamos logica de ventana movil

        for step in range(total_steps):
            # Train desde el inicio de la ventana hasta el inicio del test
            train = data.iloc[ind_test_init - train_size : ind_test_init]
            if train.shape[0] < train_size:
                raise ValueError(
                    "Tamaño incorrecto de ventana, estipula una ventana de menor tamaño"
                )

            # Test va desde el indice de inicio del test hasta sumarle el test_size
            test = data.iloc[ind_test_init : ind_test_init + test_size]

            ## Inicializamos y ajustamos el modelo a los datos de entrenamiento
            model = ExponentialSmoothing(train, **init_params)
            model = model.fit(optimized=False, **fit_params)

            # Predecimos los valores del test
            predictions = model.forecast(steps=test_size)

            if np.any(pd.isna(test)) or np.any(pd.isna(predictions)):
                return float("inf")

            # Calculamos la métrica de error
            if metric == "rmse":
                rmse = np.sqrt(mean_squared_error(test, predictions))
                metric_list.append(rmse)
            elif metric == "mae":
                mae = mean_absolute_percentage_error(test, predictions)
                metric_list.append(mae)
            else:
                raise ValueError("Métrica no reconocida. Use 'rmse' o 'mae'.")

            ind_test_init += test_size  # actualizamos el inidice inicial del test set

    elif window_type == "expanding":
        # Aplicamos logica de ventana recursiva

        for step in range(total_steps):
            # Train desde el inicio de la ventana hasta el inicio del test
            train = data.iloc[ind_test_init - train_size : ind_test_init]

            if train.shape[0] < train_size:
                raise ValueError(
                    "Tamaño incorrecto de ventana, estipula una ventana de menor tamaño"
                )

            # Test va desde el indice de inicio del test hasta sumarle el test_size
            test = data.iloc[ind_test_init : ind_test_init + test_size]

            ## Inicializamos y ajustamos el modelo a los datos de entrenamiento
            model = ExponentialSmoothing(train, **init_params)
            model = model.fit(optimized=False, **fit_params)

            # Predecimos los valores del test
            predictions = model.forecast(steps=test_size)

            if np.any(pd.isna(test)) or np.any(pd.isna(predictions)):
                return float("inf")

            if metric == "rmse":
                rmse = np.sqrt(mean_squared_error(test, predictions))
                metric_list.append(rmse)
            elif metric == "mae":
                mae = mean_absolute_percentage_error(test, predictions)
                metric_list.append(mae)
            else:
                raise ValueError("Métrica no reconocida. Use 'rmse' o 'mae'.")

            ind_test_init += test_size  # actualizamos el inidice inicial del test set
            train_size += test_size  # Aumentamos el tamaño del entrenamiento para la siguiente iteración

    else:
        raise ValueError("Tipo de ventana no reconocido. Use 'rolling' o 'expanding'.")

    metric = pd.Series(
        metric_list
    ).mean()  # Calculamos la media de las métricas obtenidas

    return metric