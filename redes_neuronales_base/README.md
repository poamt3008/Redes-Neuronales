# Implementaciones de Redes Neuronales (from Scratch)

Este directorio contiene diversas implementaciones de redes neuronales artificiales desarrolladas en MATLAB sin librerías externas. El enfoque principal es comparar los diferentes métodos de optimización y entrenamiento.

## 🧠 Algoritmos Disponibles

### 1. ANN-SGD-Scratch (Método Patrón)
Implementación utilizando **Descenso de Gradiente Estocástico (SGD)**. Los pesos se actualizan patrón por patrón, lo que introduce ruido en la convergencia pero permite salir de mínimos locales con mayor facilidad.

### 2. ANN-Batch-Scratch (Método Batch)
Implementación utilizando el **Gradiente Descendente por Lotes (Batch)**. En este script:
* Los gradientes se acumulan a lo largo de toda una época.
* La actualización de los pesos se realiza una sola vez al final de la iteración utilizando el promedio del gradiente.
* **Ventaja:** Proporciona una trayectoria de descenso más suave y estable hacia el mínimo.

## 📐 Comparativa Técnica

| Característica | Método Patrón (SGD) | Método Batch |
| :--- | :--- | :--- |
| **Actualización** | Tras cada muestra individual | Tras procesar todo el dataset |
| **Estabilidad** | Menor (oscilaciones) | Alta (trayectoria suave) |
| **Velocidad** | Convergencia rápida inicialmente | Requiere más cálculos por actualización |
| **Cómputo** | Menor uso de memoria | Requiere acumular gradientes |

## 📊 Visualización de Resultados

Ambos scripts generan:
1. **Evolución del Costo (J):** Es interesante observar cómo la curva Batch es mucho más suave (monotónica) comparada con la de SGD.
2. **Ajuste del Modelo:** Visualización de la aproximación de la red frente a los datos con ruido.

## 💻 Uso
Ejecuta cualquiera de los archivos `.m` en MATLAB para observar las diferencias en el tiempo de entrenamiento y la forma de la curva de costo.
