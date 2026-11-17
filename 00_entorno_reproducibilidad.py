# -*- coding: utf-8 -*-
"""
================================================================================
CAPTURA DEL ENTORNO COMPUTACIONAL PARA REPRODUCIBILIDAD (Tabla N1)
================================================================================
Genera un CSV con la versión de Python, el sistema operativo, el hardware y
las versiones de todas las librerías usadas en la tesis.

Uso: ejecutar en el MISMO entorno donde corriste los modelos:
    python 00_entorno_reproducibilidad.py
================================================================================
"""

import sys
import platform
import os
import csv
from datetime import datetime

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

OUTPUT_CSV = "entorno_computacional.csv"

# Librerías usadas en los scripts de la tesis
PAQUETES = [
    "numpy", "pandas", "geopandas", "shapely", "rasterio", "scipy",
    "scikit-learn", "imbalanced-learn", "matplotlib", "seaborn",
    "statsmodels", "fiona", "pyproj",
]


def main():
    filas = [
        ("Fecha de captura", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Python", sys.version.split()[0]),
        ("Sistema operativo", f"{platform.system()} {platform.release()} ({platform.version()})"),
        ("Arquitectura", platform.machine()),
        ("Procesador", platform.processor() or "no disponible"),
        ("Núcleos lógicos", os.cpu_count()),
    ]

    # RAM total (opcional, sin dependencias nuevas en Windows)
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        filas.append(("RAM total (GB)", round(stat.ullTotalPhys / 1e9, 1)))
    except Exception:
        filas.append(("RAM total (GB)", "completar manualmente"))

    print("=" * 60)
    print("ENTORNO COMPUTACIONAL")
    print("=" * 60)
    for k, v in filas:
        print(f"  {k}: {v}")

    print("\n  Versiones de librerías:")
    for paq in PAQUETES:
        try:
            v = version(paq)
        except PackageNotFoundError:
            v = "no instalada"
        filas.append((paq, v))
        print(f"    {paq}: {v}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Elemento", "Valor"])
        w.writerows(filas)

    print(f"\nGuardado: {os.path.abspath(OUTPUT_CSV)}")
    print("Copia estos valores a la Tabla N1 de la tesis (sección 4.4.8 / Anexos).")


if __name__ == "__main__":
    main()
