#!/usr/bin/env python3
from pathlib import Path
from binascii import hexlify

# ---- UPDATE THIS IF FILENAME DIFFERS ----
p = Path("~/gain_model/scripts/telemetry_bin/bad_0x314.bin").expanduser()

if not p.exists():
    raise FileNotFoundError(f"{p} not found")

b = p.read_bytes()
n = len(b)

print("===================================")
print("File:", p)
print("Total bytes:", n)
print("===================================")

# Check common packet sizes
for size in [98, 86, 100, 96, 112]:
    if n % size == 0:
        print(f"Length is multiple of {size}: {n//size} blocks")

print("\n---- First 64 bytes of file ----")
print(hexlify(b[:64]).decode())

def show_block(block_size, header_size=0):
    if len(b) < block_size:
        print(f"\nBlock size {block_size} too large for file.")
        return

    blk = b[:block_size]
    hdr = blk[:header_size]
    pay = blk[header_size:]

    print("\n===================================")
    print(f"Block size = {block_size}, Header size = {header_size}")
    print("Header (hex):", hexlify(hdr).decode())
    print("Payload first 32 bytes:", hexlify(pay[:32]).decode())
    print("Payload bytes:", len(pay))
    print("===================================")

# Hypothesis 1: Full packets (98B total, 12B header)
show_block(98, 12)

# Hypothesis 2: Payload only (86B blocks)
show_block(86, 0)
