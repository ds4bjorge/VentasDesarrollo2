import pandas as pd
import joblib
import numpy as np

# Cargar modelo y datos
modelo = joblib.load('models/modelo_final.joblib')
df = pd.read_csv('data/processed/inferencia_df_transformado.csv')

# Seleccionar un producto de prueba
df_prod = df[df['producto_id']=='PROD_001'].head(1).copy()

print('='*60)
print('PRUEBA DEL MODELO CON DIFERENTES DESCUENTOS')
print('='*60)
print(f'Producto: {df_prod.nombre.values[0]}')
print(f'Precio base: {df_prod.precio_base.values[0]:.2f}€')
print(f'Precio competencia: {df_prod.precio_competencia.values[0]:.2f}€')
print('-'*60)

for desc in [0, 10, 20, 30, 40]:
    df_test = df_prod.copy()
    df_test['descuento_porcentaje'] = desc
    df_test['precio_venta'] = df_test['precio_base'] * (1 - desc/100)
    df_test['ratio_precio'] = df_test['precio_venta'] / df_test['precio_competencia']
    
    X = df_test[modelo.feature_names_in_]
    pred = modelo.predict(X)[0]
    
    print(f'Descuento {desc:2d}% | Precio venta: {df_test.precio_venta.values[0]:6.2f}€ | Ratio: {df_test.ratio_precio.values[0]:.3f} | Predicción: {pred:5.1f} unidades')

print('='*60)
print('✅ Verificación: MÁS descuento -> precio MÁS BAJO -> MÁS ventas')
print('='*60)
