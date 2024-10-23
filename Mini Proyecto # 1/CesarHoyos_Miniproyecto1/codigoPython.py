import itertools
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from collections import Counter 
from tabulate import tabulate 

'''
Codigo base para desarrollar e implementar la teoria del modelo de Ising. 
Desarrollado por: Cesar Antonio Hoyos Pelaez. 
'''

espines = [1, -1]  # 1 representa espin arriba, -1 representa espin abajo
"""
    Visualización
"""
def generar_matrices_espines(n):
    """
        Esta función permite generar redes de espínes que solo tengan dos estados posibles
        espín arriba |1> y espín abajo |-1>.

        Arumentos: 
            parametro1 (int): define la red de espines dada por nxn. 

        Retorna: 
            list: retorna una lista que contiene 2^(nx) cada una de dimension nxn, en donde cada una de sus entradas 
                representa un micro-estado.
        
    """
    combinaciones = list(itertools.product(espines, repeat=n * n)) # Genera todas las combinaciones posibles con valores 1 y -1
    matrices = [] # Se crea una lista vacia
    for combo in combinaciones: # Recorre cada una de las combinaciones
        matriz = np.array(combo).reshape((n, n)) # Cada combinacion se redimensiona para que quede una matriz. 
        matrices.append(matriz) # Se concatena a la lista vacia
    return matrices


def energia_por_microestado(matriz): 
    """
        Esta funcion permite calcular la energía de un micro-estado aplicando condiciones de frontera
        periódicas. 

        Argumentos: 
        matriz -- matriz que representa el sistema 

        Retorna: 
        La energía del microestado.
    """
    summ = [] # Se crea una lista vacía para almacenar los términos de energía

    for j in range(0,len(matriz)): 
        # Terminos de interaccion en la frontera periodica p.b.c
        paso1 = - matriz[0,j] * matriz[-1,j] 
        summ.append(paso1) 
        paso2 = - matriz[j,0] * matriz[j,-1] 
        summ.append(paso2) 
        
        for i in range(0, len(matriz) - 1): 
            # Terminos de interaccion en el interior de la matriz f.b.c
            f1 = matriz[:,j] 
            cal1 = - f1[i] * f1[i+1] 
            summ.append(cal1)
            f2 = matriz[j,:] 
            cal2 = - f2[i] * f2[i+1]
            summ.append(cal2)

    # Se calcula la energia total sumando todos los terminos
    return sum(summ)

def tabla(matrices, energia): 
    """
        Esta función toma una lista de matrices y una lista de energías y genera una tabla
        que muestra las matrices y sus respectivas energías.

        Argumentos:
        matrices -- Una lista de matrices.
        energia -- Una lista de valores de energía correspondientes a las matrices.

        Retorna:
        Imprime una tabla que muestra las matrices y sus energías.
    """
    data = []  # Inicializamos una lista para almacenar los datos de la tabla.
    for i in range(len(energia)):
        # Recorremos las listas de matrices y energía.
        matriz_str = '['+'\n'.join(['  '.join(map(str, fila)) for fila in matrices[i]]) +  ']'
        # Agrega la matriz formateada y su energía correspondiente como una fila de datos.
        data.append([ matriz_str, energia[i]])

    # Definimos los encabezados y su energía correspondiente como una fila de datos. 
    headers = [ "Micro-estado", "Energia"]

    # Generamos la tabla utilizando la biblioteca "tabulate".
    tabla = tabulate(data, headers, tablefmt="fancy_grid")

    # Imprimimos la tabla en la consola.
    return print(tabla)

def densidad_estados(energia): 
    """
        Esta funcion calcula la densidad de estados a partir de una lista de energias. 

        Argumento: 
        energia -- una lista de energias 

        Retorna: 
        Un diccionario que contiene las energías y el número de microestados asociados a ese valor de energia. 
        Además imprime una tabla de frecuencias. 
    """
    # Utilizamos la biblioteca "Counter" para contar la frecuencia de cada energía en la lista.
    cuenta = Counter(energia) # Retorna un diccionario 

     # Extraemos las energías únicas como "Energia" y sus frecuencias como "Numero de Microestados".
    energia=cuenta.keys() # Obtenemos los valores de energia 
    num_micro_estados=cuenta.values() # Obtenemos las frecuencias

    # Definimos los encabezados de las columnas de la tabla.
    headers = ["Energia", "Numero de Microestados"]

    # Creamos un diccionario con las columnas de la tabla.
    dicc={"Energia":energia,"Numero de Microestados":num_micro_estados}

    # Imprimimos la tabla de frecuencias utilizando la biblioteca "tabulate".
    print(tabulate(dicc, headers)) 

    # Retornamos el diccionario que contiene las energías y sus frecuencias.
    return dicc 

def ajuste_grafica(dicc, energia): 
    """
        Esta función realiza un ajuste gaussiano de los datos de energía y número de microestados,
        y luego traza la curva de ajuste junto con los datos originales.

        Argumentos:
        dicc -- Un diccionario que contiene las energías y sus frecuencias.
        energia -- Una lista de energías.

        Retorna:
        Imprime un gráfico de dispersión con la curva de ajuste gaussiano y los valores ajustados
        de la media y la desviación estándar.
    """
    # Extraemos las energías y los números de microestados del diccionario.
    energias = np.array(list(dicc['Energia']) ) 
    microestados = np.array(list(dicc['Numero de Microestados']) ) 

    # Definimos la función gaussiana que será utilizada para el ajuste.
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    # Estimamos los valores iniciales para A, mu y sigma.
    A = max(microestados) # Amplitud maxima 
    mu = sum(energias) / len(energias) # Media
    sigma = np.sqrt(np.sum( (energias - mu) ** 2 ) / len(energias)) # Desviacion estandar

    # Realizamos el ajuste de la gaussiana a los datos utilizando curve_fit.
    parametros, covarianza = curve_fit(gaussian, energias, microestados, p0=[A, mu, sigma])

    # Calculamos los errores en los parámetros ajustados.
    errores_parametros = np.sqrt(np.diag(covarianza))
    error_A, error_mu, error_sigma = errores_parametros

    
    # Obtenemos los parámetros ajustados.
    A_fit, mu_fit, sigma_fit = parametros 

    # Generamos puntos para la curva de ajuste.
    x_fit = np.linspace(min(energias), max(energias), 1000)
    y_fit = gaussian(x_fit, A_fit, mu_fit, sigma_fit)

    # Implementamos el coeficiente de determinacion R^2 
    # Calculamos los residuos 
    residuos = microestados - gaussian(energias, A_fit, mu_fit, sigma_fit)

    # Calculamos la suma de los residuos al cuadrado (SSR).
    SSR = np.sum(residuos**2)

    # Calculamos la suma total de los cuadrados (SST).
    media_microestados = np.mean(microestados)
    SST = np.sum((microestados - media_microestados)**2)

     # Calculamos R^2 utilizando la fórmula.
    r_cuadrado = 1 - (SSR / SST)


    # Graficamos los datos originales y la curva de ajuste, ubicándolos en la parte superior izquierda.
    plt.scatter(energias, microestados, label='Datos originales', color='blue', alpha=0.7, marker='o')
    plt.plot(x_fit, y_fit, label='Ajuste gaussiano', color='red')

    # Agregamos anotaciones con los valores de media y desviación estándar, sin superponerse.
    plt.text(0.55, 0.95, f'Media = {mu_fit:.2f} ± {error_mu:.2f}', transform=plt.gca().transAxes, color='green')
    plt.text(0.55, 0.90, f'Desviación Estándar = {sigma_fit:.2f} ± {error_sigma:.2f}', transform=plt.gca().transAxes, color='green')
    # Mostramos el valor de R^2 en la gráfica.
    plt.text(0.55, 0.4, f'R^2 = {r_cuadrado:.2f}', transform=plt.gca().transAxes, color='purple')

    # Agregamos etiquetas para "Datos originales" y "Ajuste gaussiano" en la parte superior izquierda.
    plt.text(0.02, 0.99, 'Datos originales', transform=plt.gca().transAxes, color='blue')
    plt.text(0.02, 0.94, 'Ajuste gaussiano', transform=plt.gca().transAxes, color='red')

    plt.xlabel('Energías')
    plt.ylabel('Número de microestados')
    plt.title("Ajuste gaussiano")
    plt.grid(True)

    plt.show()


    # Imprimimos los valores ajustados de la media y la desviación estándar con sus errores.
    return print(f"Valor media {mu_fit} $\pm$ {error_mu} \nValor Desviacion Estandar {sigma_fit} $\pm$ {error_sigma}") 

 
def Z1_beta(T, energias): 
    """
        Esta función se utiliza para calcular de forma numerica la funcion de particion que recorre todas las 
        energias. 
        En este implementacion la energia debe ser para cada micro-estado. 

        Argumentos: 
            T (float) -- temperatura del sistema
            energias (list) -- Lista de energías del microestado del sistema

        Retorna: 
            list -- lista de valores acumulativos para la función partición

    """
    # Calcular el valor de beta tomando k = 1
    beta = 1/T
    # Inicializar una lista vacía para almacenar los valores acumulativos de la función de partición
    value = [] 
    # Inicializar una variable nula para el acumulado temporal
    nula = 0
    # Recorrer las energías de los microestados
    for i in energias: 
        # Calcular el término exponencial para la energía actual
        a1 = np.exp(-beta * i)
         # Actualizar el acumulado temporal
        nula = nula + a1 
        # Agregar el valor acumulado a la lista de valores
        value.append(nula)
    # Retornar la lista de valores acumulativos de la función de partición
    return value 

def Z2_beta(T, num_microestados, energias): 
    """
    Calcula la funcion de particion canonica de forma numerica para un sistema termodinamico. 

    Argumentos: 
        T (float) -- temperatura del sistema. 
        num_microestados (list) -- Lista de numeros de microestados correspondientes a la energia
        energias (list) -- lista de energias asociadas a microestado del sistema

    Retorna: 
        list -- lista de valores acumulativos de la funcion de particion 
    """

    # Calcular el valor de beta (1/T) para usarlo en los cálculos posteriores
    beta = 1/T

    # Inicializar una lista vacía para almacenar los valores acumulativos de la función de partición
    value = []

    # Inicializar una variable 'acumulado' para llevar un seguimiento acumulado de los términos exponenciales
    nula = 0

    # Recorrer los elementos de num_microestados y energias en paralelo
    for i in range(0,len(num_microestados)): 
        # Calcular el término exponencial para el microestado actual
        a1 = num_microestados[i] * np.exp( - beta * energias[i])
        # Actualizar el acumulado con el nuevo término exponencial
        nula = nula + a1 
        # Agregar el valor acumulado a la lista de valores
        value.append(nula)

    # Retornar la lista de valores acumulativos de la función de partición
    return value

def meanE_T(T, num_microestados, energias): 
    """
    Calcula la media de la energia a una temperatura T para un sistema termodinamico

    Argumentos: 
        T (float) -- temperatura a la que se calcula media de la energia
        num_microestados(list) -- lista de numeros de microestados correspondientes a la energia
        energias (list) -- lista de energias del microestado del sistema. 

    Retorna: 
        float -- media de la energia a temperatura T
    """
    
    # Inicializar listas para almacenar los valores numerador y denominador
    value_up = []
    value_down = []

    # Calcular la media de la energía utilizando la fórmula de la suma ponderada
    for i in range(0, len(num_microestados)): 
        # Calcular el término del numerador
        up = energias[i] * num_microestados[i] * np.exp(- energias[i] / T)
        value_up.append(up)
        # Calcular el término del denominador
        down = num_microestados[i] * np.exp(- energias[i] / T)
        value_down.append(down)
    # Sumar los valores del numerador y el denominador
    value_up = np.sum(value_up)
    value_down = np.sum(value_down)
    # Calcular la media de la energía dividiendo el numerador por el denominador
    return value_up/value_down 

def meanE_2(T, num_microestados, energias): 
    """
    Calcula la media de energia al cuadrado a una temperatura T para un sistema termodinamico.

    Argumentos:  
        T (float) -- temperatura a la cual se calcula la media de la energia al cuadrado 
        num_microestados (list) -- lista de numeros de microestados correspondientes a las energias 
        energias (list) -- lista de energias del microestado del sistema
    
    Retorna: 
        float -- media de la energia al cuadrado de la temperatura T. 
    """

    # Inicializar listas para almacenar los valores numerador y denominador
    value_up = []
    value_down = []

    # Calcular la media de la energía al cuadrado utilizando la fórmula de la suma ponderada
    for i in range(0, len(num_microestados)): 
        # Calcular el término del numerador
        up = energias[i]**2 * num_microestados[i] * np.exp(- energias[i] / T)
        value_up.append(up)
        # Calcular el término del denominador  
        down = num_microestados[i] * np.exp(- energias[i] / T)
        value_down.append(down)
    
    # Sumar los valores del numerador y el denominador
    value_up = np.sum(value_up)
    value_down = np.sum(value_down)

    # Calcular la media de la energía al cuadrado dividiendo el numerador por el denominador
    return value_up/value_down  

# Punto de comprobacion numerica

def izq(T, num_microestados, energias):
    """
    Calcula la suma acumulada de microestados ponderados por la exponencial negativa de energias

    Argumentos: 
        temperatura (float) -- la temperatura del sistema. 
        num_microestados (list) -- lista de número de microestados. 
        energias (list) -- lista de energias correspondientes a los microestaods
    
    Retorna: 
        float -- la suma acumulada de los microestados ponderados
    """
    
    # Inicializar una lista para almacenar los valores intermedios de la suma
    scum = []

    # Iterar a través de los microestados y energías
    for i in range(0, len(num_microestados)): 
        # Calcular el valor para cada microestado        
        up =num_microestados[i] * np.exp(- energias[i] / T)
        scum.append(up)
    
    # Calcular la suma total de los valores
    scum2 = np.sum(scum)
    return scum2  

