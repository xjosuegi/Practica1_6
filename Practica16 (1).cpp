// =============================================================================
// Practica 1.6: Algoritmo de Fuerza Bruta Distribuido con MPI y OpenMP
// Arquitectura Maestro-Esclavo. Compilar: MS-MPI + /openmp:llvm (x64)
// Solo memoria dinamica (new/delete). Sin arreglos estaticos ni std::vector.
// =============================================================================
#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <climits>
#include <chrono>
#include <thread>
#include <ctime>

// >>> CAMBIA ESTO por el nombre de tu equipo <<<
#define NOMBRE_EQUIPO "Equipo_1"

// >>> CAMBIA ESTO: nombres completos, iniciando por primer apellido, en orden alfabetico <<<
#define INTEGRANTES "Integrantes: APELLIDO1 APELLIDO2 NOMBRE(S) | APELLIDO1 APELLIDO2 NOMBRE(S) | APELLIDO1 APELLIDO2 NOMBRE(S)"

static const int TAM_MSG = 2048;

// Alfabeto base-36: digitos 0-9 (valor 0-9) seguidos de letras A-Z (valor 10-35)
static const char ALPHA[37] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// Etiquetas de mensajes MPI punto a punto usados en la busqueda distribuida
static const int TAG_DONE  = 101;  // esclavo -> maestro: rango agotado, no encontro
static const int TAG_FOUND = 102;  // esclavo -> maestro: encontro la clave
static const int TAG_STOP  = 103;  // maestro -> esclavo: detener busqueda

static const long long UMBRAL_SECUENCIAL_COMPLETO = 20000000LL; // 20 millones

// ----------------------------------------------------------------------------
// Estado global de proceso (todo en memoria dinamica)
// ----------------------------------------------------------------------------
static int    g_rank = 0, g_size = 1;
static char*  g_host    = nullptr;
static FILE*  g_log     = nullptr;
static char*  g_msg     = nullptr;
static char*  g_lin     = nullptr;
static char*  g_hora    = nullptr;
static char*  g_entrada = nullptr;
static bool   g_eof     = false;

// ----------------------------------------------------------------------------
// Registro (consola + archivo log)
// ----------------------------------------------------------------------------
static void horaActual(char* out, size_t n) {
    using namespace std::chrono;
    auto ahora = system_clock::now();
    std::time_t t = system_clock::to_time_t(ahora);
    int ms = (int)(duration_cast<milliseconds>(ahora.time_since_epoch()).count() % 1000);
    std::tm tmv;
    localtime_s(&tmv, &t);
    snprintf(out, n, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
}

static void emitirV(int hilo, int modo, const char* fmt, va_list ap) {
    #pragma omp critical(io_log)
    {
        vsnprintf(g_msg, TAM_MSG, fmt, ap);
        horaActual(g_hora, 48);
        snprintf(g_lin, TAM_MSG + 256,
                 "[%s] [Equipo: %s] [PC: %s] [Nodo MPI: %d] [Hilo OpenMP: %d] %s",
                 g_hora, NOMBRE_EQUIPO, g_host, g_rank, hilo, g_msg);
        if (g_log) { fprintf(g_log, "%s\n", g_lin); fflush(g_log); }
        if (modo == 1)      printf("%s\n", g_lin);
        else if (modo == 2) printf("%s\n", g_msg);
        fflush(stdout);
    }
}
static void registrar(int hilo, bool consola, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    emitirV(hilo, consola ? 1 : 0, fmt, ap);
    va_end(ap);
}
static void imprimir(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    emitirV(0, 2, fmt, ap);
    va_end(ap);
}

// ----------------------------------------------------------------------------
// Esperas "suaves" para el menu (los workers duermen, no giran la CPU)
// ----------------------------------------------------------------------------
static void esperar(MPI_Request* rq, bool suave) {
    if (!suave) { MPI_Wait(rq, MPI_STATUS_IGNORE); return; }
    int flag = 0;
    while (true) {
        MPI_Test(rq, &flag, MPI_STATUS_IGNORE);
        if (flag) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
static void bcastSuave(void* buf, int cnt, MPI_Datatype dt) {
    MPI_Request rq; MPI_Ibcast(buf, cnt, dt, 0, MPI_COMM_WORLD, &rq); esperar(&rq, g_rank != 0);
}

// ----------------------------------------------------------------------------
// Mapeo base-36 <-> texto, y reparto de rangos entre esclavos
// ----------------------------------------------------------------------------
static void indiceATexto(long long idx, char* out, int longitud) {
    for (int i = longitud - 1; i >= 0; --i) { out[i] = ALPHA[idx % 36]; idx /= 36; }
    out[longitud] = 0;
}
static long long pow36(int n) { long long r = 1; for (int i = 0; i < n; ++i) r *= 36; return r; }

// Reparte T combinaciones entre E esclavos (ranks 1..E). i es 1-based.
static void bloqueEsclavo(long long T, int E, int i, long long* cnt, long long* off) {
    long long base = T / E, rem = T % E; int idx = i - 1;
    *cnt = base + (idx < rem ? 1 : 0);
    *off = (long long)idx * base + (idx < rem ? idx : rem);
}

// ----------------------------------------------------------------------------
// Resultado de un esclavo y mensaje final enviado al maestro
// ----------------------------------------------------------------------------
struct MsgFin {
    long long idxGlobal;              // -1 si no se encontro
    int hilo;                         // hilo OpenMP que encontro la clave (-1 si no)
    unsigned long long evaluados;     // candidatos realmente evaluados por este esclavo
    double tiempoLocal;               // tiempo transcurrido en este esclavo
    int hilosUsados;                  // hilos OpenMP que uso realmente este esclavo
};
struct ResultadoLocal { bool encontrado; long long idxGlobal; int hiloGanador; unsigned long long evaluados; double tiempoLocal; };

// ----------------------------------------------------------------------------
// Busqueda de un esclavo: recorre [inicio, fin] en bloques (chunks). Dentro de
// cada chunk, OpenMP reparte el trabajo entre hilos con schedule(dynamic) y un
// flag compartido permite que los demas hilos salten el resto en cuanto uno
// encuentra la clave. Entre chunks (fuera de la region paralela, como exige
// MPI_THREAD_FUNNELED) el esclavo revisa con MPI_Iprobe si el maestro ya
// ordeno detenerse porque otro esclavo encontro la clave.
// ----------------------------------------------------------------------------
static ResultadoLocal ejecutarEsclavo(long long inicio, long long fin, int longitud, const char* objetivo,
                                       int T, bool detalle, unsigned long long tamChunk) {
    ResultadoLocal R{ false, -1, -1, 0, 0 };
    double t0 = MPI_Wtime();
    double ultimoLog = t0;
    unsigned long long total = (unsigned long long)(fin - inicio + 1);
    if (tamChunk > total) tamChunk = total > 0 ? total : 1;

    registrar(0, detalle, "[Inicio] Rango asignado: %lld - %lld (%llu combinaciones) | Hilos OpenMP: %d",
              inicio, fin, total, T);

    long long cursor = inicio;
    while (cursor <= fin) {
        long long iniChunk = cursor;
        long long finChunk = cursor + (long long)tamChunk - 1; if (finChunk > fin) finChunk = fin;

        // Se reparte el chunk en SUB-BLOQUES entre los hilos con schedule(dynamic). Cada iteracion del
        // "parallel for" es un sub-bloque de varios miles de candidatos (no un solo candidato): asi el costo
        // del planificador dinamico se reparte y no se convierte en el cuello de botella. El flag compartido
        // se revisa una vez por sub-bloque (parada rapida si otro hilo ya hallo la clave).
        volatile bool encontradoLocal = false; long long idxLocal = -1; int hiloLocal = -1;
        const long long SUB = detalle ? 16 : 4096;
        long long nSub = (finChunk - iniChunk + SUB) / SUB;
        #pragma omp parallel for num_threads(T) schedule(dynamic) shared(encontradoLocal, idxLocal, hiloLocal)
        for (long long b = 0; b < nSub; ++b) {
            if (encontradoLocal) continue;             // parada rapida: otro hilo ya la hallo
            long long ini = iniChunk + b * SUB;
            long long fin = ini + SUB - 1; if (fin > finChunk) fin = finChunk;
            int h = omp_get_thread_num();
            char candidato[16];
            for (long long i = ini; i <= fin; ++i) {
                indiceATexto(i, candidato, longitud);
                if (detalle)
                    registrar(h, true, "[Candidato: %s] [Estado: Evaluando]", candidato);
                if (strcmp(candidato, objetivo) == 0) {
                    #pragma omp critical(hallazgo_local)
                    {
                        if (!encontradoLocal) { idxLocal = i; hiloLocal = h; encontradoLocal = true; }
                    }
                    break;
                }
            }
        }
        R.evaluados += (unsigned long long)(finChunk - iniChunk + 1);

        if (encontradoLocal) {
            R.encontrado = true; R.idxGlobal = idxLocal; R.hiloGanador = hiloLocal;
            R.tiempoLocal = MPI_Wtime() - t0;
            return R;
        }

        // Progreso periodico en el log (no por chunk, para no saturar el archivo)
        double ahora = MPI_Wtime();
        if (!detalle && ahora - ultimoLog > 2.0) {
            registrar(0, false, "[Progreso] Evaluados %llu de %llu asignados (%.1f%%)",
                      R.evaluados, total, 100.0 * R.evaluados / (double)total);
            ultimoLog = ahora;
        }

        // Entre chunks: revisar si el maestro ya ordeno detenerse
        int flag = 0; MPI_Status st;
        MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &st);
        if (flag) {
            char dummy; MPI_Recv(&dummy, 1, MPI_BYTE, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            R.tiempoLocal = MPI_Wtime() - t0;
            return R;
        }
        cursor = finChunk + 1;
    }
    R.tiempoLocal = MPI_Wtime() - t0;
    return R;
}

// ----------------------------------------------------------------------------
// Estado de la aplicacion
// ----------------------------------------------------------------------------
struct Estado {
    char* clave = nullptr;       // clave objetivo (buffer dinamico)
    int   longitud = 0;
    long long total = 0;         // 36^longitud
    bool  claveValida = false;

    bool secuencialHecha = false, secuencialEstimado = false;
    long long idxSecuencial = -1; double tSecuencial = 0;

    bool distribuidoHecha = false, distribuidoAgotado = false;
    long long idxDistribuido = -1; double tDistribuido = 0;
    int rankGanador = -1, hiloGanador = -1, procsEsclavos = 0, hilosTotales = 0;
    long long evaluadosTotales = 0; double tasaDistribuidaEstimada = 0, tiempoEstimadoDistribuido = 0;
};

// ----------------------------------------------------------------------------
// Entrada de usuario
// ----------------------------------------------------------------------------
static long long leerEntero(const char* msg, long long def) {
    printf("%s [%lld]: ", msg, def); fflush(stdout);
    if (!fgets(g_entrada, 256, stdin)) { g_eof = true; return def; }
    char* p = g_entrada;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\n' || *p == '\r' || *p == 0) return def;
    return strtoll(p, nullptr, 10);
}

// ----------------------------------------------------------------------------
// Menu del maestro. Devuelve false si hay que re-mostrar el menu sin difundir.
// cmd[0]=opcion; el resto depende de la opcion (ver cada case).
// ----------------------------------------------------------------------------
static bool menuMaestro(Estado& s, long long* cmd) {
    printf("\n================ MENU [%s] ================\n", NOMBRE_EQUIPO);
    printf(" Procesos MPI: %d (1 maestro + %d esclavos)\n", g_size, g_size - 1);
    if (s.claveValida) printf(" Clave actual: \"%s\" (long=%d, espacio=36^%d=%lld combinaciones)\n",
                               s.clave, s.longitud, s.longitud, s.total);
    else printf(" Clave actual: (sin definir)\n");
    printf(" 1. Ingresar clave personalizada (A-Z, 0-9)\n"
           " 2. Ejecutar busqueda Secuencial (1 nodo, 1 hilo - linea base)\n"
           " 3. Ejecutar busqueda Paralela Distribuida (MPI + OpenMP)\n"
           " 4. Mostrar tabla comparativa de tiempos y Speedup\n 5. Salir\n");
    long long op = leerEntero("Opcion", 4);
    if (g_eof) op = 5;

    switch (op) {
    case 1: {
        char buf[64];
        printf("Clave a descifrar (1-10 caracteres, solo A-Z y 0-9): "); fflush(stdout);
        if (!fgets(buf, sizeof(buf), stdin)) { g_eof = true; return false; }
        size_t L = strlen(buf);
        while (L > 0 && (buf[L - 1] == '\n' || buf[L - 1] == '\r')) buf[--L] = 0;
        if (L < 1 || L > 10) { printf("Longitud invalida (debe ser 1 a 10)\n"); return false; }
        for (size_t k = 0; k < L; ++k) {
            char c = buf[k];
            if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
                printf("Caracter invalido '%c'. Solo A-Z y 0-9 en mayusculas.\n", c); return false;
            }
        }
        strncpy(s.clave, buf, 16); s.clave[16] = 0;
        cmd[0] = 1; cmd[1] = (long long)L;
        return true;
    }
    case 2: {
        if (!s.claveValida) { printf("Primero ingresa una clave (opcion 1)\n"); return false; }
        long long forzar = 0, detalle = 0;
        if (s.total > UMBRAL_SECUENCIAL_COMPLETO) {
            printf("El espacio tiene %lld combinaciones; una busqueda secuencial completa puede tardar mucho.\n", s.total);
            long long op2 = leerEntero("1=Medir con muestra y extrapolar (recomendado)  2=Forzar ejecucion COMPLETA", 1);
            forzar = (op2 == 2) ? 1 : 0;
        }
        if (s.longitud <= 3) detalle = leerEntero("Imprimir cada candidato evaluado (1=si, 0=no)", 1);
        cmd[0] = 2; cmd[1] = forzar; cmd[2] = detalle;
        return true;
    }
    case 3: {
        if (!s.claveValida) { printf("Primero ingresa una clave (opcion 1)\n"); return false; }
        if (g_size < 2) { printf("Se necesita al menos 1 proceso esclavo ademas del maestro (usa -n 2 o mas)\n"); return false; }
        long long hilos = leerEntero("Hilos OpenMP por esclavo (0=automatico)", 0);
        long long limite = 0;
        if (s.total > UMBRAL_SECUENCIAL_COMPLETO)
            limite = leerEntero("Limite de tiempo en segundos para esta prueba (0=sin limite; recomendado para N grande)", 120);
        long long detalle = (s.longitud <= 3) ? leerEntero("Imprimir cada candidato evaluado (1=si, 0=no)", 1) : 0;
        cmd[0] = 3; cmd[1] = hilos; cmd[2] = limite; cmd[3] = detalle;
        return true;
    }
    case 4: cmd[0] = 4; return true;
    case 5: cmd[0] = 5; return true;
    default: printf("Opcion invalida\n"); return false;
    }
}

static void mostrarComparativa(Estado& s) {
    if (!s.claveValida) { imprimir("Aun no hay una clave configurada."); return; }
    imprimir("");
    imprimir("================ COMPARATIVA: Fuerza Bruta (clave=\"%s\", N=%d, espacio=%lld) ================",
              s.clave, s.longitud, s.total);
    if (!s.secuencialHecha) {
        imprimir("Secuencial: (no ejecutado)");
    } else {
        imprimir("Secuencial [%s]: indice=%lld  tiempo=%.4f s%s",
                 s.secuencialEstimado ? "ESTIMADO por muestreo" : "MEDIDO completo",
                 s.idxSecuencial, s.tSecuencial,
                 s.secuencialEstimado ? "  (extrapolado; no se ejecuto completo)" : "");
    }
    if (!s.distribuidoHecha) {
        imprimir("Distribuido: (no ejecutado)");
    } else if (s.distribuidoAgotado) {
        imprimir("Distribuido: NO completado, se alcanzo el tiempo limite. Evaluados %lld de %lld (%.4f%%). Tasa combinada: %.0f candidatos/seg",
                 s.evaluadosTotales, s.total, 100.0 * s.evaluadosTotales / (double)s.total, s.tasaDistribuidaEstimada);
        imprimir("Tiempo DISTRIBUIDO ESTIMADO para agotar el espacio completo: %.2f s (%.2f horas)",
                 (double)s.tiempoEstimadoDistribuido, (double)s.tiempoEstimadoDistribuido / 3600.0);
    } else {
        imprimir("Distribuido: indice=%lld encontrado por Nodo MPI %d, Hilo OpenMP %d. Tiempo total=%.4f s. Esclavos=%d, hilos totales=%d",
                 s.idxDistribuido, s.rankGanador, s.hiloGanador, s.tDistribuido, s.procsEsclavos, s.hilosTotales);
    }
    if (s.secuencialHecha && s.distribuidoHecha) {
        double tp = s.distribuidoAgotado ? (double)s.tiempoEstimadoDistribuido : s.tDistribuido;
        int p = s.hilosTotales > 0 ? s.hilosTotales : 1;
        if (tp > 0) {
            double sp = s.tSecuencial / tp, ef = sp / p * 100.0;
            bool proyectado = s.secuencialEstimado || s.distribuidoAgotado;
            imprimir("Speedup%s = %.2f   Eficiencia%s = %.1f%% (sobre %d hilos totales: %d esclavos x %d hilos)",
                     proyectado ? " (PROYECTADO)" : "", sp, proyectado ? " (proyectada)" : "", ef,
                     p, s.procsEsclavos, s.hilosTotales / (s.procsEsclavos > 0 ? s.procsEsclavos : 1));
        }
    }
    imprimir("=========================================================================================");
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    int prov = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // PRIMERA linea de la ejecucion: nombres completos del equipo
    if (g_rank == 0) { printf("%s\n", INTEGRANTES); fflush(stdout); }

    g_host = new char[MPI_MAX_PROCESSOR_NAME + 1];
    int len = 0; MPI_Get_processor_name(g_host, &len); g_host[len] = 0;
    g_msg = new char[TAM_MSG]; g_lin = new char[TAM_MSG + 256];
    g_hora = new char[48];     g_entrada = new char[256];

    char* nombreLog = new char[600];
    snprintf(nombreLog, 600, "log_equipo_%s_nodo_%d.txt", g_host, g_rank);
    g_log = fopen(nombreLog, "a");
    if (!g_log) fprintf(stderr, "AVISO: no se pudo abrir %s\n", nombreLog);
    if (g_log) { fprintf(g_log, "%s\n", INTEGRANTES); fflush(g_log); }

    const char* cpu = getenv("PROCESSOR_IDENTIFIER");
    registrar(0, false, "=============== NUEVA SESION ===============");
    registrar(0, true,  "Inicio | Archivo log: %s | Procesos MPI: %d (1 maestro + %d esclavos) | Hilos OpenMP detectados: %d | Procesadores logicos: %d",
              nombreLog, g_size, g_size - 1, omp_get_max_threads(), omp_get_num_procs());
    registrar(0, false, "Procesador: %s", cpu ? cpu : "(desconocido)");

    Estado s; s.clave = new char[17](); // buffer de la clave (17 = 16 chars + nulo), compartido por todos los ranks
    long long* cmd = new long long[8];

    bool seguir = true;
    while (seguir) {
        for (int i = 0; i < 8; ++i) cmd[i] = 0;
        if (g_rank == 0 && !menuMaestro(s, cmd)) continue;

        bcastSuave(cmd, 8, MPI_LONG_LONG);
        long long op = cmd[0];
        registrar(0, false, "Comando recibido: opcion %lld", op);

        switch (op) {
        // ---------------------------------------------------------------- 1: ingresar clave
        case 1: {
            bcastSuave(s.clave, 17, MPI_CHAR);
            s.longitud = (int)cmd[1];
            s.total = pow36(s.longitud);
            s.claveValida = true;
            s.secuencialHecha = false; s.distribuidoHecha = false;
            if (g_rank == 0)
                imprimir("Clave configurada: \"%s\" (longitud=%d). Espacio de busqueda: 36^%d = %lld combinaciones",
                          s.clave, s.longitud, s.longitud, s.total);
            registrar(0, false, "Clave recibida: %s (long=%d, total=%lld)", s.clave, s.longitud, s.total);
            break;
        }
        // ---------------------------------------------------------------- 2: secuencial (solo nodo 0)
        case 2: {
            if (g_rank == 0) {
                bool completo = (s.total <= UMBRAL_SECUENCIAL_COMPLETO) || (cmd[1] != 0);
                bool detalle = (cmd[2] != 0);
                long long limiteMuestra = completo ? s.total : UMBRAL_SECUENCIAL_COMPLETO;
                long long idx = -1, i = 0;
                double t0 = MPI_Wtime();
                for (; i < limiteMuestra; ++i) {
                    char cand[16]; indiceATexto(i, cand, s.longitud);
                    if (detalle) registrar(0, true, "[Candidato: %s] [Estado: Evaluando]", cand);
                    if (strcmp(cand, s.clave) == 0) { idx = i; break; }
                }
                double t = MPI_Wtime() - t0;
                long long procesados = (idx >= 0) ? (idx + 1) : i;
                if (idx >= 0) {
                    s.idxSecuencial = idx; s.tSecuencial = t; s.secuencialEstimado = false; s.secuencialHecha = true;
                    imprimir("[EXITO] Busqueda secuencial: clave \"%s\" encontrada en indice %lld de %lld, tiempo=%.4f s",
                              s.clave, idx, s.total, t);
                } else if (completo) {
                    s.idxSecuencial = -1; s.tSecuencial = t; s.secuencialEstimado = false; s.secuencialHecha = true;
                    imprimir("ADVERTENCIA: no se encontro la clave en la busqueda secuencial completa (tiempo=%.4f s)", t);
                } else {
                    double tasa = (t > 0) ? (procesados / t) : 0;
                    double tEstimado = (tasa > 0) ? ((double)s.total / tasa) : -1;
                    s.idxSecuencial = -1; s.tSecuencial = tEstimado; s.secuencialEstimado = true; s.secuencialHecha = true;
                    imprimir("Muestra secuencial: %lld candidatos evaluados en %.4f s (%.0f candidatos/seg)", procesados, t, tasa);
                    imprimir("Tiempo secuencial ESTIMADO para %lld combinaciones: %.2f s (%.2f horas)",
                              s.total, tEstimado, tEstimado / 3600.0);
                }
            } else {
                registrar(0, false, "Esperando resultado de la busqueda secuencial (medicion de linea base, solo nodo 0)");
            }
            break;
        }
        // ---------------------------------------------------------------- 3: distribuida maestro-esclavo
        case 3: {
            int T = (int)cmd[1]; if (T <= 0) T = omp_get_max_threads();
            double limiteSeg = (double)cmd[2];
            bool detalle = (cmd[3] != 0);
            int E = g_size - 1;

            if (g_rank == 0) {
                bool*   yaReporto  = new bool[g_size];
                MsgFin* resultados = new MsgFin[g_size];
                for (int r = 0; r < g_size; ++r) { yaReporto[r] = false; resultados[r] = { -1, -1, 0, 0, 0 }; }
                bool encontrado = false, avisoParada = false, agotado = false;
                int rankGanador = -1, recibidos = 0;

                if (cmd[1] > 0)
                    imprimir("Iniciando busqueda distribuida: %d esclavos, %d hilos OpenMP por esclavo, espacio=%lld combinaciones",
                              E, T, s.total);
                else
                    imprimir("Iniciando busqueda distribuida: %d esclavos, hilos OpenMP AUTOMATICOS (cada esclavo usa los de su equipo), espacio=%lld combinaciones",
                              E, s.total);
                if (limiteSeg > 0) imprimir("Tiempo limite de esta prueba: %.0f s", limiteSeg);

                double t0 = MPI_Wtime();
                while (recibidos < E) {
                    int flag = 0; MPI_Status st;
                    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
                    if (flag) {
                        MsgFin m;
                        MPI_Recv(&m, sizeof(MsgFin), MPI_BYTE, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        resultados[st.MPI_SOURCE] = m; yaReporto[st.MPI_SOURCE] = true; recibidos++;
                        if (st.MPI_TAG == TAG_FOUND && !encontrado) {
                            encontrado = true; rankGanador = st.MPI_SOURCE;
                            imprimir("[EXITO] Clave \"%s\" encontrada por Nodo MPI %d, Hilo OpenMP %d, indice global %lld, tiempo parcial %.4f s",
                                      s.clave, rankGanador, m.hilo, m.idxGlobal, m.tiempoLocal);
                        } else {
                            registrar(0, detalle, "[Fin] Nodo MPI %d termino su rango sin encontrar la clave. Evaluados: %llu", st.MPI_SOURCE, m.evaluados);
                        }
                    } else if (!avisoParada && limiteSeg > 0 && (MPI_Wtime() - t0) > limiteSeg) {
                        agotado = true;
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                    if ((encontrado || agotado) && !avisoParada) {
                        avisoParada = true;
                        for (int r = 1; r < g_size; ++r)
                            if (r != rankGanador && !yaReporto[r]) { char d = 1; MPI_Send(&d, 1, MPI_BYTE, r, TAG_STOP, MPI_COMM_WORLD); }
                        if (agotado) imprimir("Tiempo limite de %.0f s alcanzado: enviando senal de parada a los esclavos pendientes", limiteSeg);
                    }
                }
                double tTotal = MPI_Wtime() - t0;
                long long evalSum = 0; int hilosReales = 0;
                for (int r = 1; r < g_size; ++r) { evalSum += (long long)resultados[r].evaluados; hilosReales += resultados[r].hilosUsados; }

                s.evaluadosTotales = evalSum; s.procsEsclavos = E; s.hilosTotales = hilosReales;
                if (encontrado) {
                    s.idxDistribuido = resultados[rankGanador].idxGlobal;
                    s.hiloGanador = resultados[rankGanador].hilo;
                    s.rankGanador = rankGanador; s.tDistribuido = tTotal; s.distribuidoAgotado = false;
                } else {
                    s.distribuidoAgotado = true;
                    double tasa = (tTotal > 0) ? (evalSum / tTotal) : 0;
                    s.tasaDistribuidaEstimada = tasa;
                    s.tiempoEstimadoDistribuido = (tasa > 0) ? (long long)((double)s.total / tasa) : -1;
                }
                s.distribuidoHecha = true;
                imprimir("Todos los procesos (%d esclavos) confirmaron su finalizacion. Tiempo total: %.4f s", E, tTotal);
                delete[] yaReporto; delete[] resultados;
            } else {
                long long cnt, off; bloqueEsclavo(s.total, E, g_rank, &cnt, &off);
                long long inicio = off, fin = off + cnt - 1;
                unsigned long long tamChunk = detalle ? 500ULL : 1000000ULL;
                ResultadoLocal R = ejecutarEsclavo(inicio, fin, s.longitud, s.clave, T, detalle, tamChunk);
                MsgFin m{ R.idxGlobal, R.hiloGanador, R.evaluados, R.tiempoLocal, T };
                if (R.encontrado) {
                    registrar(R.hiloGanador, true, "[EXITO] [Clave Encontrada: \"%s\"] [Indice Global: %lld] [Tiempo Parcial: %.4fs]",
                              s.clave, R.idxGlobal, R.tiempoLocal);
                    MPI_Send(&m, sizeof(m), MPI_BYTE, 0, TAG_FOUND, MPI_COMM_WORLD);
                } else {
                    registrar(0, true, "[Fin] Rango agotado o senal de parada recibida. Evaluados: %llu de %lld",
                              R.evaluados, cnt);
                    MPI_Send(&m, sizeof(m), MPI_BYTE, 0, TAG_DONE, MPI_COMM_WORLD);
                }
            }
            break;
        }
        // ---------------------------------------------------------------- 4: comparativa
        case 4:
            if (g_rank == 0) mostrarComparativa(s);
            break;
        // ---------------------------------------------------------------- 5: salir
        case 5:
            seguir = false;
            break;
        default: break;
        }
    }

    registrar(0, g_rank == 0, "Sesion finalizada");
    if (g_log) { fprintf(g_log, "%s\n", INTEGRANTES); fflush(g_log); }
    if (g_log) fclose(g_log);
    g_log = nullptr;
    MPI_Barrier(MPI_COMM_WORLD);       // todos terminaron: el maestro imprime la ULTIMA linea
    if (g_rank == 0) { printf("%s\n", INTEGRANTES); fflush(stdout); }

    delete[] s.clave; delete[] cmd; delete[] nombreLog;
    delete[] g_host; delete[] g_msg; delete[] g_lin; delete[] g_hora; delete[] g_entrada;
    MPI_Finalize();
    return 0;
}
