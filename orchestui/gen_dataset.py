import struct

data = [
    1.0, 2.0,
    2.0, 4.0,
    3.0, 6.0,
    4.0, 8.0,
    5.0, 10.0,
    6.0, 12.0,
    7.0, 14.0,
    8.0, 16.0,
]

with open("orchestui/data.bin", "wb") as f:
    f.write(struct.pack(f"{len(data)}f", *data))

print(f"wrote {len(data)} floats to orchestui/data.bin")
