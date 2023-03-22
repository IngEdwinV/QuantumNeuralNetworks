![Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Logo_de_la_Escuela_Colombiana_de_Ingenier%C3%ADa.svg/2560px-Logo_de_la_Escuela_Colombiana_de_Ingenier%C3%ADa.svg.png)

# QuantumNeuralNetworks 👇

### 🛠️ ¿Que herramienta Utilizamos? 

- PyCharm CE (IDE)
- [qiskit-machine-learning](https://qiskit.org/documentation/machine-learning/tutorials/01_neural_networks.html)

#

### Redes neuronales cuánticas vs. clásicas:

La redes neuronales clásicas son modelos algorítmicos basados en el cerebro humano, se basan nodos interconectados, o neuronas los cuales tiene como objetivo poder encontrar patrones en los datos y con ellos poder aprender a resolver problemas complejos, estos modelos pueden aprender con entrenamientos de aprendizaje automático o profundo

Por otro lado las QNN aplican el principio genérico de combinar redes neuronales clásicas y circuitos cuánticos parametrizados y así abrirse camino a esquemas de aprendizaje nuevos y mejorados

Las redes neuronales Quianticas se pueden ver desde dos perspectivas:

- **Aprendizaje automático** que son modelos algorítmicos que se pueden entrenar para encontrar patrones, estos pueden cargar datos clásicos en un estado cuántico y luego procesarlos con puertas cuánticas.
- **Computación cuantica** Son algoritmos cuánticos basados en circuitos cuánticos parametrizados para poderse entrenar de manera variacional utilizando optimizadores clásicos.

![test1](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test1.png)

# Implementación 🚀

La implementación de una QNN de acuerdo a la documentación y tutorial de Qiskit está pensando en unidades computacionales independientes de la aplicación y que se pueden utilizar en varios casos.

Está diseñado para implementarse de la siguiente forma:

* Una interfaz (**NeuralNetwork**) para las redes neuronales, esta será una clase abstracta de los cuales todos los QNN heredaran
* Una red (**EstimatorQNN**) basada en la evaluación de observables.
* Una red (**SamplerQNN**) basada en las muestras de medición del circuito cuántico.

### Intalación de paquetes

Se realiza la instalación de los paquetes requeridos para la ejecución del proyecto. 
- pip install 'qiskit[machine-learning] 
- pip install matplotlib 
- pip install IPython

### Crear Instancia **EstimatorQNN**:

como primer paso debemos crear la red que evaluara los observables, para esto se debe crear un circuito cuantico parametrizado.
El primer parametro sera un QNN y el otro un entrenable, teniendo el circuito podemos creal el observable y obtener como resultado lo siguiente:

![test2](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test2.png)

con el circuito y el observable podemos entonces crear el constructor EstimatorQNN y con esto la red crea un instancia primitiva local que sera la referencia para nosotros.

### Crear Instancia **SamplerQNN**:

Ahora vamos a crea la instancia que mide las mediciones resultantes de un circuito cuantico, para esto se hace de una manera similar al EstimatorQNN, pero cambiamos los parametros del circuito, en este caso tendremos dos parametros de entrada  y cuatro parametros de peso de entrenamiento, tendriamos un circuito de la siguiente manera:

![test3](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test3.png)

Con el circuito parametriza creado procedemos a crear la instancia SamplerQNN

## How to Run a Forward Pass

De acuerdo a la guia dice que en un entorno real las entradas estarian definidas por un conjunto de datos y los pesos estarian definidos por un algoritmo de entrenamiento o parte de un modelo preentrenado.

de acuerdo al tutorial se definiran las entredas como un junto de datos aleatorios y los pesos con una dimención correcta

denimos los parametros para las dos instancias:

![test4](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test4.png)


#

Ahora teniendo instanciadas y parametrizadas nuestras dos clases de QNN, vamos a ver como podemos vanazar hacia adelante en la red haciendo esto por lotes o sin lotes

De acuerdo al aguia nos dice que el paso hacia adelante sin lotes para el **EstimatorQNN** es el numero de muestra :

![test5](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test5.png)

y para **SamplerQNN** la salida es una funcion de de interpretación personalizada con la muestra:

![test6](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test6.png)

Ahora veamos el paso hacia delante por lotes

Tenemos que para **EstimatorQNN** se debe pasar el tamaño del lote, numero del Qubits y el observable:

![test7](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test7.png)

Con respecto a **SamplerQNN** es una funcion de de interpretación personalizada pero con el tamaño del lote y la muestra:

![test8](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test8.png)

#

Asi como podemos configurar que en la red se realicen pasos hacia adelante, podemos configurar para que lo haga hacia atras

De acuerdo con la guia dice que en los pasos hacia atras solo se puede calcular gradientes de acuerdo con los pesos parametrizados.


![test9](https://github.com/IngEdwinV/QuantumNeuralNetworks/blob/main/Imagenes/test9.png)


