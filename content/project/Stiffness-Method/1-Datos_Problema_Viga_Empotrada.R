
# Problema de Viga Empotrada --------------------------------------------------
# El Propósito de este código es sólo cargar la información de entrada para 
# distintos problemas planteados en la memoria.

# Creador: Alfonso Tobar --------------------------------------------------
# Fecha: 25 - 05 - 2019 ---------------------------------------------------
# Última Modificación: 25 - 05 - 2019
# Hecha por: Alfonso Tobar

library(tidyverse)
library(tibble)
library(magick)


# Datos del Problema ------------------------------------------------------

L<-1 # Largo de Barras.Para distintos largos se utiliza un factor de L.
N<-10^6 # Número de Muestras a Generar
E<-2*10^11 # Módulo de Young
A<-0.0001 # Área de Sección Transversal
N<-10^6 # Número de Muestras a Generar

# Datos del Campo Aleatorio -----------------------------------------------

mu <- E # Promedio Modulo de Young
sigma <- 2 * 10 ^ 10 # STD Módulo de Young
l <- 1 # Largo de Correlación

#Importar Imágen del Problema
# FIXME Transformar esta Imágen en SVG. Cambiar por Imágen correspondiente
# IDEA learn r2d3 to Create this using the Code below
#image_read("Codigo_2019/Problema 2 Barras.png")



# Matriz de Nodos ---------------------------------------------------------

Nodos <-
  tibble::tribble(
    ~Xi, ~Yi,
    0,   0,
    L,   0,
    2*L, 0,
    3*L, 0
  ) %>%
  as.matrix()

# Matriz de Elementos -----------------------------------------------------

Elementos <-
  tibble::tribble(
    ~ni, ~nf,  ~E,   ~A,
    1,   2,    E,    A,
    2,   3,    E,    A,
    3,   4,    E,    A
  ) %>%
  as.matrix()

Num_Nodos <- nrow(Nodos)
Num_Elementos <- nrow(Elementos)


