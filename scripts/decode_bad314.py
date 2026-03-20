#!/usr/bin/env python3
"""
decode_bad314.py

Decode 92-byte telemetry packets (6-byte wrapper + 86-byte 12-bit payload),
apply engineering conversions, write CSV, and print ALL converted values
to the shell.
"""

from bitstring import BitStream
from pathlib import Path
import math
import csv


FIELD_NAMES = [
    "THERM_FPGA", "THERM_DCB", "VMON_6V", "VMON_3V7", "VMON_1V8", "VMON_3V3D",
    "VMON_2V5D", "VMON_1V2D", "VMON_VCC_FLASH0", "VMON_VCC_FLASH1",
    "VMON_VCC_FLASH2", "VMON_VCC_FLASH3", "VMON_3V3_HK", "VMON_GND",
    "GND_RES", "GND_RES2", "SPE_P5_V", "SPE_P5_C", "SPE_N5_V", "SPE_N5_C",
    "SPE_1VA8_V", "SPE_1VA8_C", "SPE_1VAD8_V", "SPE_1VAD8_C",
    "SPE_3VD3_V", "SPE_3VD3_C", "SPE_2VD5_V", "SPE_2VD5_C",
    "SPE_1VD8_V", "SPE_1VD8_C", "SPE_1VD5_V", "SPE_1VD5_C",
    "SPE_1VD0_V", "SPE_1VD0_C", "SPE_FPGA_T", "SPE_ADC0_T", "SPE_ADC1_T",
    "PFPS_DCB_1V8", "PFPS_DCB_3V7", "PFPS_DCB_5V", "PFPS_SPE_1V5",
    "PFPS_SPE_2V3", "PFPS_SPE_3V6", "PFPS_SPE_P5V5", "PFPS_SPE_N5V5",
    "PFPS_MOT_A", "PFPS_PA3_T", "PFPS_PA2_T", "PFPS_PA1_T", "PFPS_PA0_T",
    "PFPS_CAR_T", "PFPS_PFPS_T", "PFPS_BAT_T", "VMON_PDU_COMMS",
    "VMON_PDU_PFPS", "VMON_PDU_CAROUSEL", "ADC_PWR"
]

PACKET_SIZE = 92
HEADER_SIZE = 6


# ---------------------------
# Conversion functions
# ---------------------------

def dcb_thermistor(x):
    A = 2.5
    B = 10000
    U = (4.068 / 4096) * x
    RK = (U * B / (A - U)) / 1000
    RLOG = math.log(RK)
    return round(1.303 * (RLOG ** 2) - 31.38 * RLOG + 91.44, 3)


def SPE_FPGA_T_conv(x, V=2.07):
    R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
    return round(3969 / (13.31 - math.log(10000 / R)) - 273.15, 3)


def pfps_1k_thermistor(counts):
    A = 5.010
    B = 1000
    U = (4.068 / 4096) * counts * 2
    R = (U * B / (A - U))
    return round(((R - B) / 3.85), 3)


def spec_adc_conv(x, V=2.07):
    R = (25.37 * x - 474.5) / (V + 0.04745 - 0.002537 * x)
    return round(3694 / (12.39 - math.log(10000 / R)) - 273.15, 3)


# ---------------------------
# Decode logic
# ---------------------------

def decode_file(filepath: Path):
    b = filepath.read_bytes()

    if len(b) % PACKET_SIZE != 0:
        print("Warning: file size not multiple of packet size.")

    n_packets = len(b) // PACKET_SIZE
    print(f"\nDecoding {n_packets} packet(s)\n")

    records = []

    for i in range(n_packets):
        packet = b[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
        wrapper = packet[:HEADER_SIZE]
        payload = packet[HEADER_SIZE:]

        stream = BitStream(payload)
        row = {"WRAPPER_HEX": wrapper.hex()}

        # Raw decode
        for field in FIELD_NAMES:
            row[field] = stream.read("uint:12")

        # ---------------------------
        # Apply conversions (always)
        # ---------------------------

        SPE_1VA8_V_meas = round(0.0025373 * row["SPE_1VA8_V"] - 0.0474504, 3)

        row["THERM_FPGA"] = dcb_thermistor(row["THERM_FPGA"])
        row["THERM_DCB"] = dcb_thermistor(row["THERM_DCB"])

        row["SPE_FPGA_T"] = SPE_FPGA_T_conv(row["SPE_FPGA_T"], SPE_1VA8_V_meas)
        row["SPE_ADC0_T"] = spec_adc_conv(row["SPE_ADC0_T"], SPE_1VA8_V_meas)
        row["SPE_ADC1_T"] = spec_adc_conv(row["SPE_ADC1_T"], SPE_1VA8_V_meas)

        row["SPE_P5_V"] = round(0.0134 * row["SPE_P5_V"] - 0.2505, 3)
        row["SPE_P5_C"] = round(0.0001269 * row["SPE_P5_C"] - 0.002373, 3)
        row["SPE_N5_V"] = round(-0.01021 * row["SPE_N5_V"] + 0.1908, 3)
        row["SPE_N5_C"] = round(0.0001269 * row["SPE_N5_C"] - 0.002373, 3)

        for key in ["SPE_1VA8", "SPE_1VAD8", "SPE_3VD3", "SPE_2VD5",
                    "SPE_1VD8", "SPE_1VD5", "SPE_1VD0"]:
            row[f"{key}_V"] = round(0.0025373 * row[f"{key}_V"] - 0.0474504, 3)

        for key in ["VMON_6V", "VMON_3V7", "VMON_VCC_FLASH0", "VMON_VCC_FLASH1",
                    "VMON_VCC_FLASH2", "VMON_VCC_FLASH3", "VMON_3V3_HK"]:
            row[key] = round((4.0 / 4096.0) * row[key] * 2.0, 3)

        for key in ["VMON_1V8", "VMON_2V5D", "VMON_1V2D",
                    "VMON_GND", "GND_RES", "GND_RES2"]:
            row[key] = round((4.0 / 4096.0) * row[key], 3)

        for key in ["PFPS_PA3_T", "PFPS_PA2_T", "PFPS_PA1_T", "PFPS_PA0_T"]:
            row[key] = pfps_1k_thermistor(row[key])

        row["VMON_3V3D"] = round((4.0 / 4096.0) * row["VMON_3V3D"] * 2.0, 3)

        records.append(row)

    return records


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    in_file = Path("~/gain_model/scripts/telemetry_bin/bad_0x314.bin").expanduser()
    out_csv = Path("~/gain_model/scripts/telemetry_bin/bad_0x314_decoded.csv").expanduser()

    recs = decode_file(in_file)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["WRAPPER_HEX"] + FIELD_NAMES)
        writer.writeheader()
        writer.writerows(recs)

    print(f"\nWrote {len(recs)} record(s) to {out_csv}\n")

    # ---------------------------
    # PRINT ALL converted values
    # ---------------------------

    for i, row in enumerate(recs):
        print("=" * 70)
        print(f"PACKET {i}")
        print("=" * 70)

        for key in ["WRAPPER_HEX"] + FIELD_NAMES:
            print(f"{key:>20} : {row[key]}")

        print("\n")
