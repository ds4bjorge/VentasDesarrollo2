import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .css-1d391kg {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Función para cargar datos y modelo
@st.cache_resource
def cargar_modelo_y_datos():
    """Carga el modelo y los datos de inferencia"""
    try:
        # Obtener el directorio base del proyecto (un nivel arriba de app/)
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        modelo_path = os.path.join(base_dir, 'models', 'modelo_final.joblib')
        datos_path = os.path.join(base_dir, 'data', 'processed', 'inferencia_df_transformado.csv')
        
        modelo = joblib.load(modelo_path)
        df = pd.read_csv(datos_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return modelo, df
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o datos: {str(e)}")
        return None, None

# Función para recalcular variables dependientes
def recalcular_variables(df_producto, ajuste_descuento, escenario_competencia):
    """Recalcula las variables dependientes según los controles del usuario"""
    df_sim = df_producto.copy()
    
    # Recalcular precio_venta según descuento
    # Más descuento = precio más bajo
    # Por ejemplo: descuento 20% -> precio_venta = precio_base * 0.80
    descuento_decimal = ajuste_descuento / 100
    df_sim['precio_venta'] = df_sim['precio_base'] * (1 - descuento_decimal)
    df_sim['descuento_porcentaje'] = ajuste_descuento
    
    # Ajustar precio_competencia según escenario
    if escenario_competencia == "Competencia -5%":
        factor_competencia = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_competencia = 1.05
    else:
        factor_competencia = 1.0
    
    df_sim['precio_competencia'] = df_sim['precio_competencia'] * factor_competencia
    
    # Recalcular ratio_precio
    df_sim['ratio_precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
    
    return df_sim

# Función para hacer predicciones recursivas
def predecir_recursivo(modelo, df_sim):
    """Hace predicciones día por día actualizando los lags recursivamente"""
    df_pred = df_sim.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    # Obtener las columnas que el modelo espera
    feature_cols = modelo.feature_names_in_
    
    # Lista para almacenar predicciones
    predicciones = []
    
    # Predicción recursiva día por día
    for i in range(len(df_pred)):
        # Preparar features para este día
        X = df_pred.loc[[i], feature_cols]
        
        # Hacer predicción
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # No permitir predicciones negativas
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if i < len(df_pred) - 1:
            # Desplazar lags hacia adelante
            df_pred.loc[i+1, 'lag_7'] = df_pred.loc[i, 'lag_6']
            df_pred.loc[i+1, 'lag_6'] = df_pred.loc[i, 'lag_5']
            df_pred.loc[i+1, 'lag_5'] = df_pred.loc[i, 'lag_4']
            df_pred.loc[i+1, 'lag_4'] = df_pred.loc[i, 'lag_3']
            df_pred.loc[i+1, 'lag_3'] = df_pred.loc[i, 'lag_2']
            df_pred.loc[i+1, 'lag_2'] = df_pred.loc[i, 'lag_1']
            df_pred.loc[i+1, 'lag_1'] = pred
            
            # Actualizar media móvil con las últimas 7 predicciones
            if i >= 6:
                df_pred.loc[i+1, 'media_movil_7d'] = np.mean(predicciones[i-5:i+1])
            else:
                # Para los primeros días, usar lo que tengamos
                df_pred.loc[i+1, 'media_movil_7d'] = np.mean(predicciones[:i+1])
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_predichos'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

# Función para comparar escenarios
def comparar_escenarios(modelo, df_producto, ajuste_descuento):
    """Compara los 3 escenarios de competencia"""
    escenarios = {
        "Actual (0%)": 1.0,
        "Competencia -5%": 0.95,
        "Competencia +5%": 1.05
    }
    
    resultados = {}
    
    for nombre_escenario, factor in escenarios.items():
        df_sim = df_producto.copy()
        
        # Aplicar descuento del usuario
        descuento_decimal = ajuste_descuento / 100
        df_sim['precio_venta'] = df_sim['precio_base'] * (1 - descuento_decimal)
        df_sim['descuento_porcentaje'] = ajuste_descuento
        
        # Aplicar factor de competencia
        df_sim['precio_competencia'] = df_sim['precio_competencia'] * factor
        df_sim['ratio_precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
        
        # Predecir
        df_result = predecir_recursivo(modelo, df_sim)
        
        resultados[nombre_escenario] = {
            'unidades_totales': df_result['unidades_predichas'].sum(),
            'ingresos_totales': df_result['ingresos_predichos'].sum()
        }
    
    return resultados

# Cargar modelo y datos
modelo, df_inferencia = cargar_modelo_y_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# Lista de productos únicos
productos = sorted(df_inferencia['nombre'].unique())

# ===== SIDEBAR =====
st.sidebar.markdown("## 🎮 Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "🏷️ **Selecciona un Producto:**",
    productos,
    index=0
)

# Slider de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 **Porcentaje de Descuento:**",
    min_value=0,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Aplica un descuento al precio base del producto. Más descuento = precio más bajo = más ventas esperadas"
)

# Selector de escenario de competencia
escenario_competencia = st.sidebar.radio(
    "🏪 **Escenario de Competencia:**",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Simula cambios en los precios de la competencia"
)

st.sidebar.markdown("---")

# Botón de simular
simular = st.sidebar.button("🚀 **SIMULAR VENTAS**", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Información")
st.sidebar.info("""
**Instrucciones:**
1. Selecciona un producto
2. Ajusta el descuento deseado
3. Elige un escenario de competencia
4. Haz clic en 'Simular Ventas'

**Nota:** Las predicciones se calculan día por día actualizando los lags recursivamente.
""")

# ===== ZONA PRINCIPAL =====
st.markdown("# 📊 Dashboard de Simulación de Ventas")
st.markdown(f"### Noviembre 2025 - {producto_seleccionado}")
st.markdown("---")

if simular:
    with st.spinner("⏳ Generando predicciones recursivas..."):
        # Filtrar datos para el producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        # Recalcular variables según controles
        df_simulado = recalcular_variables(df_producto, ajuste_descuento, escenario_competencia)
        
        # Hacer predicciones recursivas
        df_resultado = predecir_recursivo(modelo, df_simulado)
        
        # ===== KPIs DESTACADOS =====
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unidades_totales = int(df_resultado['unidades_predichas'].sum())
            st.metric(
                label="📦 Unidades Totales",
                value=f"{unidades_totales:,}",
                delta=None
            )
        
        with col2:
            ingresos_totales = df_resultado['ingresos_predichos'].sum()
            st.metric(
                label="💶 Ingresos Proyectados",
                value=f"{ingresos_totales:,.2f} €",
                delta=None
            )
        
        with col3:
            precio_promedio = df_resultado['precio_venta'].mean()
            st.metric(
                label="💵 Precio Promedio Venta",
                value=f"{precio_promedio:.2f} €",
                delta=None
            )
        
        with col4:
            descuento_promedio = df_resultado['descuento_porcentaje'].mean()
            st.metric(
                label="🎯 Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # ===== GRÁFICO DE PREDICCIÓN DIARIA =====
        st.markdown("### 📈 Predicción Diaria de Unidades Vendidas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Configurar estilo seaborn
        sns.set_style("whitegrid")
        
        # Gráfico de línea principal
        sns.lineplot(
            data=df_resultado,
            x='dia_mes',
            y='unidades_predichas',
            marker='o',
            markersize=6,
            linewidth=2.5,
            color='#667eea',
            ax=ax
        )
        
        # Marcar Black Friday (día 28)
        black_friday_data = df_resultado[df_resultado['dia_mes'] == 28]
        if not black_friday_data.empty:
            bf_unidades = black_friday_data['unidades_predichas'].values[0]
            ax.plot(28, bf_unidades, 'ro', markersize=15, label='Black Friday', zorder=5)
            ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.annotate(
                f'🛍️ Black Friday\n{bf_unidades:.0f} unidades',
                xy=(28, bf_unidades),
                xytext=(28, bf_unidades + max(df_resultado['unidades_predichas']) * 0.1),
                ha='center',
                fontsize=11,
                fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2)
            )
        
        ax.set_xlabel('Día de Noviembre', fontsize=13, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas Predichas', fontsize=13, fontweight='bold')
        ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 31))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # ===== TABLA DETALLADA =====
        st.markdown("### 📋 Detalle Diario de Predicciones")
        
        # Preparar tabla
        df_tabla = df_resultado[['fecha', 'dia_mes', 'nombre_dia_semana', 'precio_venta', 
                                  'precio_competencia', 'descuento_porcentaje', 
                                  'unidades_predichas', 'ingresos_predichos']].copy()
        
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%Y-%m-%d')
        df_tabla['precio_venta'] = df_tabla['precio_venta'].round(2)
        df_tabla['precio_competencia'] = df_tabla['precio_competencia'].round(2)
        df_tabla['descuento_porcentaje'] = df_tabla['descuento_porcentaje'].round(1)
        df_tabla['unidades_predichas'] = df_tabla['unidades_predichas'].round(0).astype(int)
        df_tabla['ingresos_predichos'] = df_tabla['ingresos_predichos'].round(2)
        
        # Añadir emoji de Black Friday
        df_tabla['fecha'] = df_tabla.apply(
            lambda row: f"🛍️ {row['fecha']}" if row['dia_mes'] == 28 else row['fecha'],
            axis=1
        )
        
        df_tabla = df_tabla.rename(columns={
            'fecha': 'Fecha',
            'dia_mes': 'Día',
            'nombre_dia_semana': 'Día Semana',
            'precio_venta': 'Precio Venta (€)',
            'precio_competencia': 'Precio Competencia (€)',
            'descuento_porcentaje': 'Descuento (%)',
            'unidades_predichas': 'Unidades',
            'ingresos_predichos': 'Ingresos (€)'
        })
        
        st.dataframe(df_tabla, height=400)
        
        st.markdown("---")
        
        # ===== COMPARATIVA DE ESCENARIOS =====
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        
        with st.spinner("📊 Comparando escenarios..."):
            resultados_escenarios = comparar_escenarios(modelo, df_producto, ajuste_descuento)
        
        col1, col2, col3 = st.columns(3)
        
        escenarios_lista = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        colores = ["#667eea", "#52c41a", "#fa8c16"]
        
        for col, escenario, color in zip([col1, col2, col3], escenarios_lista, colores):
            with col:
                st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: white; margin: 0;'>{escenario}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: white; margin: 10px 0;'>{int(resultados_escenarios[escenario]['unidades_totales']):,}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: white; margin: 0;'>unidades</p>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: white; margin: 10px 0;'>{resultados_escenarios[escenario]['ingresos_totales']:,.2f} €</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: white; margin: 0;'>ingresos</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Mensaje de éxito
        st.success("✅ Simulación completada exitosamente!")

else:
    # Pantalla de bienvenida
    st.info("👈 Configura los parámetros en el panel lateral y haz clic en '🚀 SIMULAR VENTAS' para comenzar.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Características de la Simulación")
        st.markdown("""
        - ✅ **Predicciones recursivas** día por día
        - ✅ **Actualización automática de lags** con predicciones anteriores
        - ✅ **Black Friday** destacado visualmente
        - ✅ **Comparativa de escenarios** de competencia
        - ✅ **KPIs en tiempo real**
        - ✅ **Tabla detallada** con toda la información
        """)
    
    with col2:
        st.markdown("### 📊 Datos del Modelo")
        st.markdown(f"""
        - **Productos disponibles:** {len(productos)}
        - **Período de predicción:** Noviembre 2025 (30 días)
        - **Modelo:** HistGradientBoostingRegressor
        - **Variables incluidas:** Temporales, lags, precios, competencia
        - **Datos con lags inicializados** desde octubre
        """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>💡 Desarrollado con ❤️ para optimización de ventas</p>", unsafe_allow_html=True)
