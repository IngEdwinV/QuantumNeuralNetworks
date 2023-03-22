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
