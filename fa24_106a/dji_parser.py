import re
from datetime import datetime

def parse_metadata(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by subtitle blocks
    blocks = content.strip().split('\n\n')
    # print(blocks) 
    # EX) '1\n00:00:00,000 --> 00:00:00,033\n<font size="28">FrameCnt: 1, 
    # DiffTime: 33ms\n2025-05-07 16:02:57.632\n[iso: 100] [shutter: 1/1750.36] 
    # [fnum: 1.7] [ev: 0] [color_md : default] [focal_len: 24.00] [latitude: 37.871293]
    #  [longitude: -122.317558] [rel_alt: 106.500 abs_alt: 71.130] [ct: 5025] </font>',
    results = []

    for block in blocks:
        lines = block.strip().split('\n')
        # detects
        # print(lines)
        # if len(lines) < 4:
        #     continue

        frame_num = int(lines[0].strip())
        timestamp_range = lines[1].strip()
        metadata_line = '\n'.join(lines[2:]).strip()
        # print(metadata_line)
        
        # Extract timestamp info
        start_time, end_time = timestamp_range.split(' --> ')

        # Parse frame count and diff time
        framecnt_match = re.search(r'FrameCnt: (\d+), DiffTime: (\d+)ms', metadata_line)
        if framecnt_match:
            framecnt = int(framecnt_match.group(1))
            difftime = int(framecnt_match.group(2))
        else:
            continue
        # print(framecnt, difftime)

        # Parse datetime
        datetime_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', metadata_line)
        timestamp = datetime.strptime(datetime_match.group(1), "%Y-%m-%d %H:%M:%S.%f") if datetime_match else None

        # Extract key-value fields in brackets
        kv_matches = re.findall(r'\[(.*?)\]', metadata_line)
        # print(kv_matches)
        kv_data = {}
        for kv in kv_matches:
            if 'rel_alt' in kv and 'abs_alt' in kv:
                match = re.search(r'rel_alt: ([\d.]+) abs_alt: ([\d.]+)', kv)
                if match:
                    kv_data['rel_alt'] = float(match.group(1))
                    kv_data['abs_alt'] = float(match.group(2))
            elif ':' in kv:
                key, val = kv.split(':', 1)
                try:
                    kv_data[key.strip()] = float(val.strip())
                except ValueError:
                    kv_data[key.strip()] = val.strip()


        # Convert types where relevant
        float_fields = ['iso', 'shutter', 'fnum', 'ev', 'focal_len', 'latitude', 'longitude', 'ct']
        for k in float_fields:
            if k in kv_data:
                try:
                    kv_data[k] = float(kv_data[k])
                except:
                    pass
        print()

        results.append({
            'frame': framecnt,
            'diff_ms': difftime,
            'start_time': start_time,
            'end_time': end_time,
            'datetime': timestamp,
            **kv_data
        })

    return results

# Example usage
parsed = parse_metadata("DJI_20250507160257_0026_D.SRT")

# Optional: print or save as CSV
import pandas as pd
df = pd.DataFrame(parsed)
df.to_csv("parsed_output.csv", index=False)
print(df.head())