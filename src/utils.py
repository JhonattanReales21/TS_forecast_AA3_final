import plotly.graph_objects as go


def graficar_serie_con_descomposicion(df, titulo, nombre_columna):
    """
    Función para graficar una serie de tiempo con su descomposición en tendencia y estacionalidad.
    """
    df_plot = df.reset_index().copy()

    # Crear figura
    fig = go.Figure()

    # Pedidos (eje primario)
    fig.add_trace(
        go.Scatter(
            x=df_plot["fecha"],
            y=df_plot[nombre_columna],
            mode="lines",
            name="Pedidos",
            line=dict(width=2),
            yaxis="y1",
        )
    )

    # Tendencia (eje secundario)
    fig.add_trace(
        go.Scatter(
            x=df_plot["fecha"],
            y=df_plot[f"{nombre_columna}_trend"],
            mode="lines",
            name="Tendencia",
            line=dict(width=2, dash="dash"),
            yaxis="y2",
        )
    )

    # Estacionalidad (eje secundario)
    fig.add_trace(
        go.Scatter(
            x=df_plot["fecha"],
            y=df_plot[f"{nombre_columna}_seasonal"],
            mode="lines",
            name="Estacionalidad",
            line=dict(width=2, dash="dot"),
            yaxis="y2",
        )
    )

    # Layout del gráfico
    fig.update_layout(
        title=f"Serie de tiempo de ventas para {titulo} con Tendencia y Estacionalidad",
        xaxis=dict(title="Fecha", showgrid=True, gridcolor="LightGray"),
        yaxis=dict(title="Ventas", showgrid=True, gridcolor="LightGray"),
        yaxis2=dict(
            title="Tendencia / Estacionalidad",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0.01, y=0.99),
        template="simple_white",
    )

    fig.show()
