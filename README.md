# Practica1_6 Fuerza bruta distribuida con MPI y OpenMP

Programa que busca por fuerza bruta una clave de 1 a 10 caracteres (letras mayúsculas A-Z y dígitos 0-9). El espacio de búsqueda tiene 36^N combinaciones para una clave de N caracteres.

El programa usa una arquitectura maestro-esclavo:

El rango 0 es el maestro. Coordina el menú, reparte los rangos y recoge resultados. Casi no consume CPU.
Los esclavos (rangos 1 en adelante) reciben un bloque del espacio de búsqueda. Dentro de cada esclavo, OpenMP reparte los candidatos entre sus hilos.
Archivos
Archivo	Descripción
Practica16_corregida.cpp	Código fuente. Es la versión a usar.
log_equipo_<PC>_nodo_<rango>.txt	Se genera al ejecutar. Un archivo por proceso, con todo lo que se imprime.
Requisitos
Windows de 64 bits.
Visual Studio 2019 versión 16.9 o posterior (necesario para /openmp:llvm).
MS-MPI (SDK y runtime) instalado en todas las máquinas.
En cada esclavo, el servicio smpd de MS-MPI debe estar activo.
La DLL de OpenMP de LLVM (normalmente libomp140.x86_64.dll, que está en VC\Tools\Llvm\x64\bin de Visual Studio) debe estar junto al .exe en cada máquina.
Compilación

Abre el x64 Native Tools Command Prompt for VS y ejecuta:

cl /EHsc /O2 /openmp:llvm Practica16_corregida.cpp /I"%MSMPI_INC%" /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib

Antes de entregar, cambia NOMBRE_EQUIPO e INTEGRANTES al inicio del archivo.

Ejecución en una sola máquina
mpiexec -n 2 -wdir C:\clustermpi C:\clustermpi\Practica16_corregida.exe
-n 2: número de procesos MPI. Son 1 maestro y 1 esclavo. Con -n 1 solo funciona la búsqueda secuencial.
-wdir <carpeta>: carpeta de trabajo de cada proceso. Ahí se crean los archivos de log.
Después de -wdir va la ruta del ejecutable.
Ejecución en dos máquinas

Ejemplo con la máquina 192.168.0.10 (6 núcleos) y la 192.168.0.11 (8 núcleos):

mpiexec -hosts 2 192.168.0.10 2 192.168.0.11 1 -wdir C:\clustermpi C:\clustermpi\Practica16_corregida.exe

El formato de -hosts es:

-hosts <número de máquinas> <IP1> <procesos en IP1> <IP2> <procesos en IP2> ...

En el ejemplo, 192.168.0.10 lanza 2 procesos (maestro y esclavo) y 192.168.0.11 lanza 1 proceso (esclavo). En total hay 3 procesos y 2 esclavos.

Antes de la primera ejecución:

Copia el .exe y la DLL de OpenMP a la misma ruta local en ambas máquinas (por ejemplo C:\clustermpi). No uses rutas de red.
Usa el mismo usuario y contraseña en ambas máquinas.
Permite en el firewall de ambas máquinas los programas smpd.exe, mpiexec.exe y Practica16_corregida.exe.
Verifica la conexión:
   mpiexec -hosts 2 192.168.0.10 1 192.168.0.11 1 hostname

Deben aparecer los dos nombres de equipo.

Menú del programa

Al iniciar, el maestro muestra el menú:

Opción	Qué hace
1	Ingresa la clave a buscar (1 a 10 caracteres, solo A-Z y 0-9 en mayúsculas).
2	Búsqueda secuencial en el maestro, con 1 hilo. Es la línea base. Si el espacio tiene más de 20 millones de candidatos, pregunta si quieres una muestra con extrapolación (1) o la ejecución completa (2).
3	Búsqueda distribuida (MPI + OpenMP). Pide los hilos por esclavo, un límite de tiempo y si se imprimen los candidatos.
4	Muestra la tabla comparativa de tiempos, speedup y eficiencia.
5	Sale del programa.

En la opción 3:

Hilos OpenMP por esclavo:
0 (automático): cada esclavo usa todos sus procesadores lógicos.
N: cada esclavo usa N hilos. El mismo valor se aplica a todos los esclavos.
Límite de tiempo: en segundos. Si se alcanza, el maestro detiene a los esclavos y estima el tiempo total. Usa 0 para no tener límite.
Imprimir candidatos: muestra cada candidato evaluado. Solo para claves cortas, porque genera mucha salida.
Configuraciones sugeridas para comparar
Configuración	Comando	Menú (opción 3): hilos	Hilos de cálculo
Secuencial	(opción 2)	(no aplica)	1
5 esclavos, 1 hilo	-n 6	1	5
6 esclavos, 1 hilo (núcleos de .10)	-n 7	1	6
12 esclavos, 1 hilo (procesadores lógicos de .10)	-n 13	1	12
Híbrido, 2 procesos × 6 hilos	-n 2	6	6
Dos máquinas, híbrido	-hosts 2 192.168.0.10 2 192.168.0.11 1	0	12 + 16 lógicos

Para una comparación justa, usa la misma clave en todas las pruebas, preferiblemente una que esté al final del espacio (por ejemplo ZZZZZZ), para que todas recorran el espacio completo. En la opción 2, responde 2 para la ejecución completa.

Cómo leer los resultados
Speedup = tiempo secuencial / tiempo paralelo.
Eficiencia = speedup / hilos de cálculo.
Si el equipo tiene hyperthreading, la eficiencia sobre hilos lógicos sale más baja de lo que refleja el hardware. Calcula también la eficiencia sobre núcleos físicos.
El programa reporta los hilos totales que usó cada esclavo y calcula la tabla con esos valores.
Solución de problemas
"Se necesita al menos 1 proceso esclavo": lanza el programa con -n 2 o más, o con -hosts con al menos 2 procesos.
Error por la DLL de OpenMP al ejecutar: copia libomp140.x86_64.dll junto al .exe en esa máquina.
Un esclavo no responde o el programa se queda esperando: revisa que smpd esté activo en el esclavo, que el firewall permita los programas y que el usuario y la contraseña coincidan.
Con más hilos el programa tarda más: dentro de cada proceso, OpenMP tiene que coordinar los hilos en cada candidato. Con 1 hilo por proceso, el reparto entre procesos suele dar mejores tiempos. El ajuste schedule(dynamic, 4096) reduce esa coordinación.
Limitaciones conocidas
El reparto entre esclavos es proporcional a sus hilos, pero no considera diferencias de velocidad por hilo entre máquinas.
Los nombres del equipo y de los integrantes deben cambiarse antes de entregar, porque el programa los imprime al inicio y al final.
