import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Dict
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
import warnings

def evalua_modelo_ST_por_ventana(
    index_name: str,
    model_type: str,
    init_params: Dict,
    fit_params: Dict,
    data: pd.DataFrame,
    test_init: str,
    test_finish: str,
    window_type: str = "rolling",
    train_size: int = 60,
    test_size: int = 1,
    metric: str = "rmse",
):
    """
    Utiliza una ventana móvil o recursiva para evaluar un modelo ETS, promedio_movil o prophet en un periodo de prueba específico.
    Esta función solo es valida si la data es mensual, ya que se basa en el tamaño de la ventana en meses.

    Parámetros:
    - index_name: Nombre del índice de la serie temporal (ej. 'fecha').
    - model_type: Tipo de modelo a evaluar ('ETS','MA','RLM', 'prophet').
    - init_params: Parámetros iniciales del modelo.
    - fit_params: Parámetros de ajuste del modelo.
    - data: DataFrame con la serie temporal a evaluar.
    - test_init: Fecha de inicio del periodo de prueba (formato 'YYYY-MM-DD').
    - test_finish: Fecha de fin del periodo de prueba (formato 'YYYY-MM-DD').
    - window_type: Tipo de ventana a utilizar ('rolling' o 'expanding').
    - train_size: Tamaño de la ventana de entrenamiento (en meses).
    - test_size: Tamaño de la ventana de prueba (en meses).
    - metric: Métrica a utilizar para evaluar el modelo ('rmse' o 'mape').
    """

    metric_list = []
    temp_data = data.reset_index().copy()
    ind_test_init = temp_data[temp_data[index_name] == test_init].index[0]

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
            # Train va hasta el inicio del test
            train = data.iloc[ind_test_init - train_size : ind_test_init]
            if train.shape[0] < train_size:
                raise ValueError(
                    "Tamaño incorrecto de ventana, estipula una ventana de menor tamaño"
                )

            # Test va desde el indice de inicio del test hasta sumarle el test_size
            test = data.iloc[ind_test_init : ind_test_init + test_size]

            if model_type == "ETS":
                ## Inicializamos y ajustamos el modelo a los datos de entrenamiento
                model = ExponentialSmoothing(train, **init_params)
                model = model.fit(optimized=False, **fit_params)

                # Predecimos los valores del test
                predictions = model.forecast(steps=test_size)

            elif model_type == "prophet":
                pass

            elif model_type == "MA":
                mov_avg = train.rolling(window=fit_params["window_size"]).mean()
                last_mov_avg = mov_avg.dropna().iloc[-1]
                predictions = pd.Series(
                    [last_mov_avg] * fit_params["horizon"], index=test.index
                )

            else:
                raise ValueError("Modelo no reconocido. Use 'ETS' o 'prophet'.")

            if np.any(pd.isna(test)) or np.any(pd.isna(predictions)):
                return float("inf")

            # Calculamos la métrica de error
            if metric == "rmse":
                rmse = np.sqrt(mean_squared_error(test, predictions))
                metric_list.append(rmse)
            elif metric == "mape":
                mape = mean_absolute_percentage_error(test, predictions)
                metric_list.append(mape)
            else:
                raise ValueError("Métrica no reconocida. Use 'rmse' o 'mape'.")

            ind_test_init += test_size  # actualizamos el indice inicial del test set

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

            if model_type == "ETS":
                ## Inicializamos y ajustamos el modelo a los datos de entrenamiento
                model = ExponentialSmoothing(train, **init_params)
                model = model.fit(optimized=False, **fit_params)

                # Predecimos los valores del test
                predictions = model.forecast(steps=test_size)

            elif model_type == "prophet":
                pass
            
            elif model_type == "MA":
                mov_avg = train.rolling(window=fit_params["window_size"]).mean()
                last_mov_avg = mov_avg.dropna().iloc[-1]
                predictions = pd.Series(
                    [last_mov_avg] * fit_params["horizon"], index=test.index
                )

            else:
                raise ValueError("Modelo no reconocido. Use 'ETS' o 'prophet'.")

            if np.any(pd.isna(test)) or np.any(pd.isna(predictions)):
                return float("inf")

            if metric == "rmse":
                rmse = np.sqrt(mean_squared_error(test, predictions))
                metric_list.append(rmse)
            elif metric == "mape":
                mape = mean_absolute_percentage_error(test, predictions)
                metric_list.append(mape)
            else:
                raise ValueError("Métrica no reconocida. Use 'rmse' o 'mape'.")

            ind_test_init += test_size  # actualizamos el inidice inicial del test set
            train_size += test_size  # Aumentamos el tamaño del entrenamiento para la siguiente iteración

    else:
        raise ValueError("Tipo de ventana no reconocido. Use 'rolling' o 'expanding'.")

    metric = pd.Series(
        metric_list
    ).mean()  # Calculamos la media de las métricas obtenidas

    return metric


def optimizar_modelo_ets(
    data,
    test_init,
    test_finish,
    window_type,
    train_size=60,
    test_size=1,
    metric="rmse",
    opt_trial=100,
):
    """
    Optimiza un modelo ETS utilizando Optuna para encontrar los mejores hiperparámetros.

    parametros:
    - data: Serie temporal a evaluar (DataFrame).
    - test_init: Fecha de inicio del periodo de prueba (formato 'YYYY-MM-DD').
    - test_finish: Fecha de fin del periodo de prueba (formato 'YYYY-MM-DD').
    - window_type: Tipo de ventana a utilizar ('rolling' o 'expanding').
    - train_size: Tamaño de la ventana de entrenamiento (en meses).
    - test_size: Tamaño de la ventana de prueba (en meses).
    - metric: Métrica a utilizar para evaluar el modelo ('rmse' o 'mae').
    - opt_trial: Número de iteraciones de Optuna para optimizar los hiperparámetros.
    """

    def objective(trial):
        ## Definimos los hiperparámetros a optimizar
        trend = trial.suggest_categorical(
            "trend", ["add", "mul", None]
        )  # Parametro de tendencia
        seasonal = trial.suggest_categorical(
            "seasonal", ["add", "mul", None]
        )  # Parametro de estacionalidad
        use_box_cox = trial.suggest_categorical(
            "use_box_cox", [True, False]
        )  # Parametro de Box-Cox
        slevel = trial.suggest_float(
            "smoothing_level", 0.001, 1.0, step=0.001
        )  # valor de suavización del nivel

        strend = None
        damped_trend = False
        damping_value = None
        seasonal_periods = None
        sseasonal = None

        if trend is not None:
            strend = trial.suggest_float(
                "smoothing_trend", 0.001, 1.0, step=0.001
            )  # Valor de suavización de la tendencia
            damped_trend = trial.suggest_categorical(
                "damped_trend", [True, False]
            )  # Condicional de amortiguar la tendencia

            if damped_trend:
                damping_value = trial.suggest_float(
                    "damping_trend", 0.01, 0.99, step=0.01
                )  # Valor de amortiguación de la tendencia

        if seasonal is not None:
            seasonal_periods = trial.suggest_int(
                "seasonal_periods", 4, 24, step=1
            )  # periodos de estacionalidad, por ejemplo, 12 para datos mensuales
            sseasonal = trial.suggest_float(
                "smoothing_seasonal", 0.001, 1.0, step=0.001
            )  # Valor de suavización de la estacionalidad

        # creamos el diccionario de parametros para inicializar el modelo ETS
        init_params = {
            "trend": trend,
            "damped_trend": damped_trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "use_boxcox": use_box_cox,
        }

        # Creamos el diccionario de parametros para ajustar el modelo ETS
        fit_params = {
            "smoothing_level": slevel,
            "smoothing_trend": strend,
            "smoothing_seasonal": sseasonal,
            "damping_trend": damping_value,
        }

        # Evaluamos el modelo utilizando la función de evaluación por ventana
        window_metric = evalua_modelo_ST_por_ventana(
            index_name="fecha",
            model_type="ETS",
            init_params=init_params,
            fit_params=fit_params,
            data=data,
            test_init=test_init,
            test_finish=test_finish,
            window_type=window_type,
            train_size=train_size,
            test_size=test_size,
            metric=metric,

        )

        return window_metric

    # Creamos el estudio de Optuna y optimizamos la función objetivo
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=opt_trial)
    best_params = study.best_params
    print(f"Mejor resultado: {study.best_value} con parámetros: {best_params}")

    return study, best_params, study.best_value


def ajustar_modelos_arima(
    train,
    test,
    p_range=range(0, 10),
    d_range=range(0, 3),
    q_range=range(0, 10),
    alpha=0.05
):
    """
    Ajusta modelos ARIMA(p,d,q) para p in p_range, d in d_range, q in q_range
    (saltando el caso p=d=q=0), calcula RMSE, y testa sus residuos:
      - Ruido blanco (Ljung-Box, lag=10)
      - Normalidad (Jarque-Bera)
      - Homoscedasticidad (ARCH, nlags=12)
      - Ruido blanco en residuos² (Ljung-Box, lag=10)

    Devuelve un DataFrame con:
      Modelo, RMSE, MAPE, p-values de cada prueba, y CumpleSupuestos (bool).
    """
    resultados = []

    def _rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    

                    # 1) Ajuste y forecast
                    model  = ARIMA(train, order=(p, d, q))
                    result = model.fit()
                    fc     = result.forecast(steps=len(test))
                    rmse_v = _rmse(test, fc)

                    # 2) Residuos
                    resid = result.resid.dropna()[1:]  # Eliminar NaN/0 inicial

                    # 3) Ljung-Box sobre resid
                    lb_p    = acorr_ljungbox(resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
                    # 4) Jarque-Bera
                    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
                    # 5) ARCH (heteroscedasticidad)
                    arch_p  = het_arch(resid, nlags=12)[1]
                    # 6) Ljung-Box sobre resid²
                    lb2_p   = acorr_ljungbox(resid**2, lags=[10], return_df=True)['lb_pvalue'].iloc[0]

                    cumple = all(pval > alpha for pval in (lb_p, jb_p, arch_p, lb2_p))

                    resultados.append({
                        "Modelo":          f"ARIMA({p},{d},{q})",
                        "RMSE":            rmse_v,
                        "LB10_p":          lb_p,
                        "JB_p":            jb_p,
                        "ARCH12_p":        arch_p,
                        "LB2_10_p":        lb2_p,
                        "CumpleSupuestos": cumple
                    })

                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) falló: {e}")
                    continue

    df = pd.DataFrame(resultados)
    # Ordena primero los que cumplen supuestos, luego por RMSE ascendente
    df = df.sort_values(by=["CumpleSupuestos", "RMSE"], ascending=[False, True])
    return df.reset_index(drop=True)

