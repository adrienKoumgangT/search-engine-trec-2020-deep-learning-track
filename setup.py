import platform
import sys
import os
import psutil


USE_STEMMING = True
USE_STOPWORDS = False



def print_runtime_characteristics():
    """Prints a summary of the current Python and hardware environment."""
    print("=" * 50)
    print("       RUNTIME ENVIRONMENT REPORT")
    print("=" * 50)

    # --- Python Information ---
    print("\n--- Python ---")
    print(f"Version:      {sys.version.split()[0]}")
    print(f"Implementation: {platform.python_implementation()}")
    print(f"Executable:   {sys.executable}")

    # --- Operating System ---
    print("\n--- Operating System ---")
    print(f"System:       {platform.system()} {platform.release()}")
    print(f"Machine:      {platform.machine()}")
    print(f"Platform:     {platform.platform()}")

    # --- CPU Information ---
    print("\n--- CPU ---")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical Cores:  {psutil.cpu_count(logical=True)}")
    # The line below attempts to get the CPU brand string, but may fail on some systems
    try:
        import cpuinfo
        print(f"Processor:      {cpuinfo.get_cpu_info()['brand_raw']}")
    except ImportError:
        pass  # cpuinfo is optional

    # --- Memory (RAM) ---
    print("\n--- Memory ---")
    svmem = psutil.virtual_memory()
    # Helper to format bytes into readable units (GB)
    def bytes_to_gb(bytes_val):
        return f"{bytes_val / (1024**3):.2f} GB"

    print(f"Total RAM:     {bytes_to_gb(svmem.total)}")
    print(f"Available:     {bytes_to_gb(svmem.available)}")
    print(f"Used:          {bytes_to_gb(svmem.used)} ({svmem.percent}%)")

    print("\n" + "=" * 50)

