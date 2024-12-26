# Overview de los tipos de capa que componen la red neuronal

## [Convolución 2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
### Parámetros:
- *in_channels*: Número de canales de entrada de la imagen de entrada (ej: 3 para RGB, 1 para escala de grises).
- *stride*: Determina cuántos píxeles "salta" el kernel al deslizarse sobre la imagen:
    - stride=1: Se mueve pixel por pixel (default).
    - stride=2: Salta de 2 en 2 píxeles, reduciendo las dimensiones espaciales a la mitad.
    - stride>2: Reduce aún más las dimensiones, pero puede perder información.
- *padding*: Añade píxeles alrededor de la imagen antes de aplicar la convolución:
    - Sin padding (padding=0): La imagen resultante se reduce.
    - padding='same': Mantiene las dimensiones de entrada iguales.
    - padding=1: Añade 1 píxel de relleno en todos los bordes.
    - padding=(1,2): Añade diferente padding vertical (1) y horizontal (2).
   
   Por defecto estos píxeles añadidos son ceros (padding_mode='zeros').
- *kernel_size*: Define el tamaño de la ventana que se desliza sobre la imagen para realizar la convolución:
    - kernel_size=3: Ventana de 3x3 píxeles.
    - kernel_size=5: Ventana de 5x5 píxeles.
    - kernel_size=(3,5): Ventana rectangular (3 alto x 5 ancho).
- *out_channels (filters)*: Determina cuántos filtros diferentes aplicará la capa convolucional. Cada filtro aprenderá a detectar un patrón diferente (bordes, texturas, formas, etc).

### La fórmula para calcular las dimensiones de salida es:
Output Height = [(Input Height + 2 × Padding - Kernel Size) / Stride] + 1

Output Width = [(Input Width + 2 × Padding - Kernel Size) / Stride] + 1

Output Channels = out_channels

### Requisito para que la convolución sea válida
Tanto el ancho como el alto de la imágen de entrada (usemos la variable *Size*) deben cumplir lo siguiente:

Size ≥ (Kernel Size - 2×Padding - 1) × Stride + 1

## [Convolución 2D Traspuesta](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
### Parámetros:
Funciona de forma similar a la convolución 2D pero "al revés". Tiene los mismos parámetros, salvo que se agrega además el siguiente:
- *output_padding*: Padding adicional para la salida.

### La fórmula para calcular las dimensiones de salida es:
Output Height = (Input Height - 1) × Stride - 2 × Padding + Kernel Size + Output Padding

Output Width = (Input Width - 1) × Stride - 2 × Padding + Kernel Size + Output Padding

Output Channels = out_channels

### Requisito para que la convolución sea válida
Tanto el ancho como el alto de la imágen de entrada (usemos la variable *Size*) deben cumplir lo siguiente:

Size ≥ (1 + 2×Padding - Kernel Size - Output Padding) / Stride + 1

## [Max Pooling](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
### Parámetros:
- *kernel_size*: Tamaño de la ventana de pooling.
- *stride*: Determina cuántos píxeles "salta" el kernel al deslizarse sobre la imagen. Por defecto es igual a kernel_size.
- *padding*: Cantidad de padding (default 0).

### La fórmula para calcular las dimensiones de salida es:
Output Height = [(Input Height + 2 × Padding - Kernel Size) / Stride] + 1

Output Width = [(Input Width + 2 × Padding - Kernel Size) / Stride] + 1

Output Channels = Input Channels

### Requisito para que la convolución sea válida
Size ≥ (Kernel Size - 2×Padding - 1) × Stride + 1