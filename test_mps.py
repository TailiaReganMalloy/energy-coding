import os
import torch


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def get_system_memory():
    total = None
    available = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        total = page_size * total_pages
        if "SC_AVPHYS_PAGES" in os.sysconf_names:
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            available = page_size * avail_pages
    except (ValueError, OSError, AttributeError):
        pass
    return total, available


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)

    if hasattr(torch.mps, "current_allocated_memory"):
        alloc = torch.mps.current_allocated_memory()
        print(f"MPS current allocated: {bytes_to_gib(alloc):.2f} GiB")
    if hasattr(torch.mps, "driver_allocated_memory"):
        driver_alloc = torch.mps.driver_allocated_memory()
        print(f"MPS driver allocated: {bytes_to_gib(driver_alloc):.2f} GiB")

    total_mem, avail_mem = get_system_memory()
    if total_mem is not None:
        print(f"System total memory: {bytes_to_gib(total_mem):.2f} GiB")
    if avail_mem is not None:
        print(f"System available memory: {bytes_to_gib(avail_mem):.2f} GiB")
        if "driver_alloc" in locals():
            est_free = max(avail_mem - driver_alloc, 0)
            print(f"Estimated free for MPS (approx): {bytes_to_gib(est_free):.2f} GiB")
else:
    print("MPS device not found.")