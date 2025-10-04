import struct
import sys

def read_ply(filename):
    """Read PLY file and return list of points (x, y, z)."""
    points = []
    
    with open(filename, 'rb') as f:
        # Read header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        format_type = None
        vertex_count = 0
        header_end = False
        
        while not header_end:
            line = f.readline().decode('ascii').strip()
            
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line == 'end_header':
                header_end = True
        
        # Read vertices based on format
        if format_type == 'binary_little_endian':
            for _ in range(vertex_count):
                x, y, z = struct.unpack('<fff', f.read(12))
                points.append((x, y, z))
        elif format_type == 'ascii':
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                coords = line.split()
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                points.append((x, y, z))
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    return points

def morton_encode(x, y, z):
    """Compute Morton code (Z-order) for given coordinates."""
    def split_by_3(a):
        """Spread bits of a by inserting two zeros between each bit."""
        x = a & 0x1fffff  # Only consider first 21 bits
        x = (x | x << 32) & 0x1f00000000ffff
        x = (x | x << 16) & 0x1f0000ff0000ff
        x = (x | x << 8) & 0x100f00f00f00f00f
        x = (x | x << 4) & 0x10c30c30c30c30c3
        x = (x | x << 2) & 0x1249249249249249
        return x
    
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)

def normalize_coords(points):
    """Normalize coordinates to integer range for Morton encoding."""
    # Find bounding box
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    min_z = min(p[2] for p in points)
    max_z = max(p[2] for p in points)
    
    # Calculate ranges
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    
    # Handle case where all points are at same coordinate
    if range_x == 0:
        range_x = 1.0
    if range_y == 0:
        range_y = 1.0
    if range_z == 0:
        range_z = 1.0
    
    # Normalize to [0, 2^21-1] range
    scale = 2**21 - 1
    normalized = []
    
    for x, y, z in points:
        # Handle NaN or inf values
        if not (float('-inf') < x < float('inf')):
            x = min_x
        if not (float('-inf') < y < float('inf')):
            y = min_y
        if not (float('-inf') < z < float('inf')):
            z = min_z
            
        nx = int((x - min_x) / range_x * scale)
        ny = int((y - min_y) / range_y * scale)
        nz = int((z - min_z) / range_z * scale)
        
        # Clamp to valid range
        nx = max(0, min(scale, nx))
        ny = max(0, min(scale, ny))
        nz = max(0, min(scale, nz))
        
        normalized.append((nx, ny, nz))
    
    return normalized

def check_morton_order(points):
    """Check if points are in Morton order."""
    normalized = normalize_coords(points)
    morton_codes = [morton_encode(x, y, z) for x, y, z in normalized]
    
    # Check if Morton codes are in non-decreasing order
    is_sorted = all(morton_codes[i] <= morton_codes[i+1] 
                   for i in range(len(morton_codes)-1))
    
    # Calculate statistics
    total = len(morton_codes)
    out_of_order = sum(1 for i in range(len(morton_codes)-1) 
                      if morton_codes[i] > morton_codes[i+1])
    
    return is_sorted, out_of_order, total, morton_codes

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <ply_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    print(f"Reading PLY file: {filename}")
    points = read_ply(filename)
    print(f"Loaded {len(points)} points")
    
    print("\nChecking Morton order...")
    is_sorted, out_of_order, total, morton_codes = check_morton_order(points)
    
    print(f"\nResults:")
    print(f"  Total points: {total}")
    print(f"  In Morton order: {'YES' if is_sorted else 'NO'}")
    print(f"  Out-of-order pairs: {out_of_order} ({out_of_order/(total-1)*100:.2f}%)")
    
    if not is_sorted:
        print(f"\nFirst few out-of-order positions:")
        count = 0
        for i in range(len(morton_codes)-1):
            if morton_codes[i] > morton_codes[i+1]:
                print(f"  Position {i}: code={morton_codes[i]} > code={morton_codes[i+1]}")
                count += 1
                if count >= 5:
                    break

if __name__ == "__main__":
    main()