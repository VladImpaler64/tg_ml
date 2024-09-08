# Clasifik - A CLI utility

#### Clasificador multiclase basado en perceptrón (OvA)

Clasifik te ayuda a evaluar los post que publicas en Telegram.

#### ¿Cómo funciona?

Lo básico a entender es que se trata de responder a una pregunta de investigación
en este caso lo que se intenta responder es ¿A qué categoría pertenece este post de Telegram?, para responder
esta pregunta me apoyo de el algoritmo del perceptrón, variante multiclase (OvA - One vs All) donde se 
evalua la entrada con las cinco clases (Excelente, Bueno, Regular, Malo, Deficiente) basado en los siguientes parámetros numéricos:

- Suscriptores: La cantidad de suscriptores del canal, ej. 200 subs
- Comentarios Totales del Post: Cuántos comentarios generó el post, ej. 23 comentarios
- Comentarios Congruentes: Cuántos comentarios se relacionan con el tema (los sticker no cuentan), ej. 15 comentarios 
- Usuarios Distintos: Cuántos usuarios distintos comentaron, ej. 6 personas
- Cantidad de Palabras: Cuántas palabras tenía el post, ej. 54 palabras
- Horario (24hrs): El horario de publicación de post, en formato 24 hrs(sin minutos), ej. 14 (2 de la tarde)
- Minutos Primer Comentario: Cuántos minutos pasaron para el primer comentario: ej. 3 minutos
- Reacciones Positivas: Número de reacciones positivas, ej. 23
- Reacciones Negativas: Número de reacciones negativas, ej. 3
- Tema actual: ¿El post trata de un tema actual?, binario si (1), no (0), ej 1 el post sí es sobre tema actual
- Encuesta: ¿Se agregó una encuesta en los comentarios?, binario, ej. 0 no se agregó
- Estrellas: Número de estrellas (el token para dar propinas en Telegram), ej. 3 obtuvo tres estrellas 
- Tipo de Post: Para este punto se debe juzgar en base a las siguientes categorías, ej 4 es un post personal
> Nota: Se cambia esto por un hot encoding, se agregan cinco parámetros más representando el tipo de post(binario), quedaría 
como 0, 0, 0, 1, 0, presencia de tipo 4, personal

### Tipo de post

1. Informativo:

Noticias: Artículos, enlaces o resúmenes de noticias relevantes para tu audiencia.
Tutoriales: Instrucciones paso a paso sobre cómo hacer algo.
Consejos: Consejos útiles o prácticos sobre diferentes temas.
Datos curiosos: Información interesante o poco conocida.
Reviews: Opiniones sobre productos, servicios o experiencias.

2. Entretenimiento:

Memes: Imágenes o videos humorísticos.
Chistes: Cuentos o bromas divertidas.
Videos virales: Contenido popular y compartido masivamente.
Música: Canciones, listas de reproducción o recomendaciones musicales.
Memes: Imágenes o videos humorísticos.

3. Interacción:

Preguntas: Preguntas abiertas para fomentar la conversación.
Encuestas: Cuestionarios para recopilar opiniones.
Concursos: Actividades con premios para incentivar la participación.
Desafíos: Retos divertidos o creativos para los usuarios.
Debates: Discusiones sobre temas controvertidos.

4. Personal:

Vlogs: Diarios en video sobre tu vida personal o profesional.
Historias personales: Relatos de experiencias personales.
Reflexiones: Pensamientos o ideas sobre diferentes temas.
Recomendaciones: Sugerencias de libros, películas, música, etc.

5. Comunitario:

Anuncios: Información importante sobre el canal o la comunidad.
Eventos: Promociones de eventos o reuniones.
Colaboraciones: Presentaciones de otros canales o usuarios.
Agradecimientos: Reconocimiento a los miembros activos de la comunidad.

- Hashtag: ¿Tiene hashtags?, ej. **0** No tiene ningún hashtag
> Añadimos algo de feature engineering
- Correlación Usuarios_Canal:Comentarios, cantidad, ej. trunc(200 / 23) **8**
- Correlación Usuarios_Canal:Usuarios_Distintos, ej. trunc(200 / 6) **33**

> Nota: El órden de los parámetros es importante y que estén los 14 parámetros, si
no se tiene información dejarlo en 0, ej. si no hay comentarios

De lo anterior obtenemos el input(14 parámetros) para el modelo, separado por comas (tipo csv) 
    `200,23,15,6,54,14,2,23,3,1,0,0,0,3,0,0,4,8,33,0`
## Comando para usarlo

Antes que nada debes hacer un:
```sh 
git clone https://github.com/VladImpaler64/tg_ml.git
cd tg_ml
```
Luego tener instalado python para correr el comando
`python main.py`
> Nota: En el futuro voy a mejorar esta interfaz y separar el código de entrenamiento
para que se pueda entrenar otro tipo de clases.

Para este ejemplo el modelo predice el siguiente output: 

```sh 
Clasificando la entrada [200, 23, 15, 6, 54, 14, 2, 23, 3, 1, 0, 3, 4, 0, 0]
[np.float64(-725.803122355756), np.float64(-80.40312235576822), np.float64(-12696.943122356033), np.float64(-895.9631223557977), np.float64(-2556.363122355685)]
The classification of this post is: Bueno
```
La primer línea nos repite el input, la segunda nos dice la probabilidad que obtuvo
en cada etiqueta de clase, la última es la clasificación, a continuación pongo qué significaría:

1 - Excelente:
Viral: Contenido que se comparte masivamente y genera muchas reacciones.
Innovador: Publicaciones que presentan ideas o enfoques nuevos y originales.
Inspiracional: Contenido que motiva y emociona a la audiencia.
Autoritativo: Publicaciones respaldadas por expertos o fuentes confiables.

2 - Bueno:
Relevante: Contenido que está directamente relacionado con los intereses de la audiencia.
Entretenido: Publicaciones que divierten y generan momentos agradables.
Útil: Contenido que proporciona soluciones a problemas o preguntas comunes.
Interactivo: Publicaciones que fomentan la participación y el debate.

3 - Regular:
Aceptable: Contenido que cumple los requisitos básicos pero no destaca.
Mejorable: Publicaciones que podrían ser mejores con algunas modificaciones.
Genérico: Contenido que es demasiado común o poco específico.

4 - Malo:
Irrelevante: Contenido que no tiene nada que ver con el tema del canal.
Repetitivo: Publicaciones que se han compartido anteriormente o son muy similares.
Ofensivo: Contenido que es inapropiado, discriminatorio o violento.
Erróneo: Publicaciones que contienen información falsa o inexacta.

5 - Deficiente:
Baja calidad: Contenido con errores ortográficos, gramaticales o de formato.
Desactualizado: Publicaciones que ya no son relevantes debido a que la información es antigua.
Poco atractivo: Contenido visualmente poco atractivo o con un diseño pobre.

### Consideraciones
- El modelo ya está entrenado, si se desea entrenar agregar la flag **--train <path>**
> Esto queda pendiente
- Los archivos que contienen los pesos están númerados por cada clase, en la carpeta pretrained
- Algunas entradas al modelo dependen de juicio subjetivo por 
la naturaleza de la pregunta de investigación -> ¿A qué categoría pertenece mi post?
- El modelo tiene no converge en la categoría de Regular, lo que significa que podría clasificar
mal para arriba (Bueno) o abajo(Malo), quizás me faltan más datos para entrenarlos con esta clase
si quieren aportar con esos datos son bien recibidos, ja
