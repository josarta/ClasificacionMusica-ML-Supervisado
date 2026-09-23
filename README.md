# Clasificación de Música por Género mediante Machine Learning Supervisado
**Examen Final — Septiembre de 2026**  
**Programa:** Maestría en Inteligencia Artificial / Ciencia de Datos  
**Curso:** Machine Learning Supervisado  

---

## ¿De qué trata este proyecto?

En este repositorio presento mi entrega del examen final del curso de **Machine Learning Supervisado**. El reto consiste en construir un sistema capaz de predecir a qué género musical pertenece una canción utilizando únicamente sus características acústicas y timbrales (extraídas del *Million Song Dataset* de Columbia LabROSA).

El enunciado del examen nos pide clasificar únicamente **5 géneros**:
1. **Dance and electronic** (`dance and electronica` en el dataset)
2. **Jazz and blues**
3. **Soul and reggae**
4. **Punk**
5. **Metal**

Para que el modelo realmente aprenda patrones musicales y no memorice "atajos" (*data leakage* o atajos por popularidad), se filtraron todos los demás géneros y se eliminaron por completo los metadatos identificatorios: el `track_id`, el título de la canción y el nombre del artista. La clasificación se basa al 100% en las **30 características numéricas de audio** restantes (dinámicas temporales, tonalidad y descriptores espectrales de timbre).

---

## Enfoque y Metodología

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
   * Evalué 4 familias representativas de algoritmos supervisados con diferente sesgo inductivo:
     * Regresión Logística Multinomial ($L_2$)
     * Random Forest Classifier
     * Extreme Gradient Boosting (XGBoost)
     * Red Neuronal Perceptrón Multicapa (MLP)
   * Diagnostiqué la velocidad de cómputo y la brecha de sobreajuste (*overfitting*), identificando a **XGBoost** como el clasificador más certero y eficiente (71.67% de Macro F1 en validación en tan solo 1.39 segundos).

3. **Selección y Regularización (20%):**
   * Unifiqué los conjuntos de entrenamiento y validación (80% del total, 15,799 muestras) para maximizar el aprendizaje en validación cruzada.
   * Utilicé **Validación Cruzada Estratificada de 5 particiones (5-Fold Stratified CV)** con `scoring='f1_macro'`.
   * Realicé búsqueda en malla con `GridSearchCV` sobre parámetros de regularización estocástica y estructural de XGBoost (`subsample: 0.8`, `max_depth: 6`, `n_estimators: 150`, `learning_rate: 0.1`), logrando un salto cuantitativo de **71.67% a 73.01%** en Macro F1.

4. **Calidad del Modelo y Desempeño Futuro (20%):**
   * **Criterio de desempeño justificado:** Elegí **Macro F1-Score** y **Balanced Accuracy** como métricas principales debido al desbalance entre clases (2.35:1).
   * **Evaluación en prueba ciega final ($N = 2,789$):**
     * **Exactitud Global (Accuracy):** **73.54%**
     * **Exactitud Balanceada (Balanced Acc):** **73.23%**
     * **Macro F1-Score:** **73.94%**
     * **Macro-promedio ROC AUC:** **0.9318** (Metal: 0.9663, Jazz: 0.9408, Punk: 0.9281, Soul: 0.9193, Dance: 0.9033).
   * **¿Qué esperamos en datos futuros?** Apliqué **Bootstrapping no paramétrico con 1,000 réplicas** en el conjunto de prueba para construir intervalos de confianza al 95%:
     * **Macro F1 esperado:** **73.95%** (IC 95%: [72.34%, 75.54%], error estándar: 0.0084).
     * **Balanced Accuracy esperada:** **73.25%** (IC 95%: [71.60%, 74.92%], error estándar: 0.0087).
     * **Accuracy global esperada:** **73.57%** (IC 95%: [71.96%, 75.19%], error estándar: 0.0081).

5. **Análisis de Resultados y Música (20%):**
   * **¿Qué géneros se confunden más en la Matriz de Confusión?**
     * **Metal vs Punk (13.3%):** Confusión mutua por compartir guitarras saturadas de alta ganancia, compases de 4/4 acelerados y máxima compresión dinámica.
     * **Jazz/Blues vs Soul/Reggae (~10%):** Confusión por su raíz común en la música negra, instrumentación de vientos (saxos, trompetas) y calidez armónica.
     * **Dance vs Soul/Reggae (~12.5%):** Por el uso intensivo de samples vocales y ritmos funk/reggae en la música electrónica de club.
     * **Polos opuestos:** La confusión entre Metal y Jazz es de apenas 1.9%, confirmando la coherencia musical del modelo.
   * **Variables más críticas (Permutación en Test):**
     * `avg_timbre6` (Balance y amplitud espectral): **-0.0949** en Macro F1.
     * `duration` (Duración de la pista): **-0.0744** en Macro F1 (canciones breves de Punk vs temas extensos de Jazz/Dance).
     * `avg_timbre5` (Armónicos centrales): **-0.0492**.
     * `loudness` (Volumen / compresión): **-0.0478**.
     * **El mito del ritmo:** El `tempo` (BPM) apenas pesó un 3.8% y no entró al Top 8, demostrando que el género musical lo define el timbre instrumental y no la velocidad.

---

## Estructura del Repositorio

```text
├── context/
│   └── Examen_Final.pdf               # Documento con las pautas y rúbrica del examen
├── data/
│   └── genre_dataset.txt              # Dataset crudo original del Million Song Dataset
├── Examen_Final.ipynb                 # Jupyter Notebook final ejecutado (entregable principal)
└── README.md                          # Este documento explicativo
```

---

## ¿Cómo reproducir los experimentos?

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

## Conclusiones Personales

Trabajar con este dataset muestra que clasificar audio musical a nivel de pistas completas usando solo promedios y varianzas de timbre es un problema complejo: el límite de rendimiento para modelos clásicos ronda el 73-75% de Macro F1. Esto se debe a que la música no vive en "cajas cerradas"; muchos temas son fusiones (como punk-metal o jazz-funk). Para un paso siguiente en producción, sería muy interesante trabajar con espectrogramas de Mel en dos dimensiones y arquitecturas convolucionales o de atención (Transformers), o bien plantear el problema como clasificación multietiqueta (*multi-label*).

