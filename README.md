# Clasificación de Música por Género mediante Machine Learning Supervisado
**Examen Final — Septiembre de 2026**  
**Programa:** Maestría en Inteligencia Artificial / Ciencia de Datos  
**Curso:** Machine Learning Supervisado  

---

## 🎵 ¿De qué trata este proyecto?

En este repositorio presento mi entrega del examen final del curso de **Machine Learning Supervisado**. El reto consiste en construir un sistema capaz de predecir a qué género musical pertenece una canción utilizando únicamente sus características acústicas y timbrales (extraídas del *Million Song Dataset* de Columbia LabROSA).

El enunciado del examen nos pide clasificar únicamente **5 géneros**:
1. **Dance and electronic** (`dance and electronica` en el dataset)
2. **Jazz and blues**
3. **Soul and reggae**
4. **Punk**
5. **Metal**

Para que el modelo realmente aprenda patrones musicales y no memorice "atajos" (*data leakage* o atajos por popularidad), se filtraron todos los demás géneros y se eliminaron por completo los metadatos identificatorios: el `track_id`, el título de la canción y el nombre del artista. La clasificación se basa al 100% en las **30 características numéricas de audio** restantes (dinámicas temporales, tonalidad y descriptores espectrales de timbre).

---

## 💡 Enfoque y Metodología

Para cumplir con todos los criterios de evaluación de la rúbrica, estructuré el trabajo de la siguiente manera:

1. **Preprocesamiento y Limpieza (10%):**
   * Cargué el archivo omitiendo los metadatos y comentarios iniciales.
   * Revisé nulos y duplicados (el subconjunto filtrado cuenta con 18,588 canciones limpias sin valores faltantes).
   * Particioné los datos en tres bloques disjuntos con **muestreo estratificado**:
     * **Entrenamiento (70%):** 13,010 canciones para ajustar los modelos.
     * **Validación (15%):** 2,789 canciones para comparar familias de modelos y ajustar hiperparámetros.
     * **Prueba (15%):** 2,789 canciones como conjunto ciego final (*holdout*) que simula datos futuros.
   * Apliqué estandarización (`StandardScaler`) ajustando los parámetros **únicamente** con el conjunto de entrenamiento para evitar fuga de información (*data leakage*).

2. **Entrenamiento de Modelos y Diagnóstico (10%):**
   * Evalué 6 modelos de diferentes familias supervisadas:
     * Regresión Logística Multinomial ($L_2$)
     * Linear Support Vector Classifier (LinearSVC)
     * Random Forest
     * Extra Trees
     * HistGradientBoosting (GBDT basado en histogramas)
     * Red Neuronal Perceptrón Multicapa (MLP)
   * Generé **Curvas de Aprendizaje (*Learning Curves*)** para verificar que los modelos no tuvieran problemas de subajuste (*underfitting*) y que la varianza estuviera controlada al aumentar las muestras de entrenamiento.

3. **Selección y Regularización (20%):**
   * Utilicé **Validación Cruzada Estratificada de 5 particiones (5-Fold CV)** sobre el conjunto de entrenamiento.
   * Analicé el impacto de los parámetros de regularización con **Curvas de Validación (*Validation Curves*)** (penalización $L_2$ de hojas en Gradient Boosting, profundidad máxima en Random Forest e inversa de regularización $C$ en modelos lineales).
   * Construí un **Ensamble de Votación Suave (*Soft-Voting Ensemble*)** que combina las probabilidades predichas por HistGradientBoosting, Random Forest y Extra Trees, logrando la menor pérdida logarítmica y el mejor rendimiento global.

4. **Calidad del Modelo y Desempeño Futuro (20%):**
   * **Criterio de desempeño justificado:** Elegí **Macro F1-Score** y **Balanced Accuracy** como métricas principales. Como existe un desbalance moderado entre géneros (desde 2,103 canciones en Metal hasta 4,935 en Dance), el Macro F1 trata a todas las clases con la misma importancia y evita que un modelo "tramposo" obtenga un falso buen puntaje a costa de descuidar los géneros minoritarios.
   * **¿Qué esperamos en datos futuros?** Evalué el modelo en el conjunto de prueba ciego y apliqué **Bootstrapping no paramétrico con 1,000 réplicas** para calcular un intervalo de confianza al 95%:
     * **Macro F1 esperado:** **73.28%** (IC 95%: [71.62%, 74.88%])
     * **Balanced Accuracy esperada:** **73.54%** (IC 95%: [71.89%, 75.14%])
     * **Accuracy global esperada:** **73.11%** (IC 95%: [71.49%, 74.72%])

5. **Análisis de Resultados y Música (20%):**
   * **¿Qué géneros se confunden más?**
     * **Metal vs Punk:** Tienen una confusión mutua de alrededor del 12-15%. Tiene todo el sentido acústico: ambos usan guitarras con distorsión pesada (armónicos altos saturados en los timbres 2 y 4), ritmos rápidos en compás de 4/4 y niveles de volumen muy comprimidos (`loudness` entre -6 y -8 dB).
     * **Jazz/Blues vs Soul/Reggae:** Comparten un 10-14% de confusión debido a sus raíces afroamericanas compartidas, instrumentación acústica (vientos, metales, pianos eléctricos cálidos) y un rango dinámico más amplio que la música comercial moderna.
     * **Dance/Electronica:** Es el género más fácil de identificar (~81% de F1), ya que sus bombos sintetizados continuos de baja frecuencia y su tempo perfectamente cuantizado generan una firma tímbrica única.
   * **Variables más importantes:** A través de *Permutation Importance*, encontré que la sonoridad global (`loudness`), la energía y brillo tímbrico (`avg_timbre1`, `avg_timbre2`), y la variabilidad dinámica (`var_timbre1`) son las que más mueven la aguja a la hora de predecir.

---

## 📁 Estructura del Repositorio

```text
├── context/
│   └── Examen_Final.pdf               # Documento con las pautas y rúbrica del examen
├── data/
│   └── genre_dataset.txt              # Dataset crudo original del Million Song Dataset
├── Examen_Final.ipynb                 # Jupyter Notebook final ejecutado (entregable principal)
└── README.md                          # Este documento explicativo
```

---

## 🚀 ¿Cómo reproducir los experimentos?

Todo el código está pensado para ejecutarse de forma limpia y reproducible en Python 3.10 o superior.

### 1. Requisitos
Las librerías requeridas son las habituales del ecosistema científico de Python:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter nbformat nbclient
```

### 2. Abrir y revisar el Notebook
El informe completo con todas sus salidas, gráficas y tablas ya se encuentra 100% ejecutado en:
* **`Examen_Final.ipynb`**

Puedes abrirlo directamente con Jupyter Notebook o VS Code:
```bash
jupyter notebook Examen_Final.ipynb
```

### 3. (Opcional) Reejecutar todo de cero
Si deseas reentrenar todas las familias de modelos, recalcular la validación cruzada y regenerar las 1,000 réplicas de bootstrapping desde la terminal:
```bash
python generate_exam_notebook.py
```
*(El proceso toma aproximadamente 3 minutos en una máquina convencional gracias a la paralelización con `n_jobs=-1`)*.

---

## 📝 Conclusiones Personales

Trabajar con este dataset muestra que clasificar audio musical a nivel de pistas completas usando solo promedios y varianzas de timbre es un problema complejo: el límite de rendimiento para modelos clásicos ronda el 73-75% de Macro F1. Esto se debe a que la música no vive en "cajas cerradas"; muchos temas son fusiones (como punk-metal o jazz-funk). Para un paso siguiente en producción, sería muy interesante trabajar con espectrogramas de Mel en dos dimensiones y arquitecturas convolucionales o de atención (Transformers), o bien plantear el problema como clasificación multietiqueta (*multi-label*).

