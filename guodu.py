line_httpcode_total = line['httpcode_total']

if '200' in line_httpcode_total:
    httpcode_total_200.append(int(line_httpcode_total['200']))
else:
    httpcode_total_200.append(int(0))

if '302' in line_httpcode_total:
    httpcode_total_302.append(int(line_httpcode_total['302']))
else:
    httpcode_total_302.append(int(0))

if '404' in line_httpcode_total:
    httpcode_total_404.append(int(line_httpcode_total['404']))
else:
    httpcode_total_404.append(int(0))

if '403' in line_httpcode_total:
    httpcode_total_403.append(int(line_httpcode_total['403']))
else:
    httpcode_total_403.append(int(0))

if '500' in line_httpcode_total:
    httpcode_total_500.append(int(line_httpcode_total['500']))
else:
    httpcode_total_500.append(int(0))