import re
import sys
import os
import glob
import gzip
import argparse
from collections import defaultdict
from datetime import datetime

"""
- Parses logs produced by NGINX
- Generates ranked table of IP addresses, showing request counts, HTTP method breakdown and date range of activity
- Filtering, can be chained:
    * --ip for history of a specific endpoint including url paths, parameters and UA's
    * --code to isolate behaviours like IP's generating 404 errors or only successul 200 reqs 
"""

LOG_DIR = "/var/log/nginx/"
LOG_FILENAME_PATTERN = "access.log*"

# --- ROBUST REGEX ---
# 1. IP (Any non-whitespace)
# 2. Date (Anything inside brackets)
# 3. Request (Anything inside quotes)
# 4. Status (3 digits)
# 5. Bytes (Digits or dash)
# 6. Referrer (Anything inside quotes)
# 7. UA (Anything inside quotes)
LOG_PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<date>.*?)\] "(?P<request_str>.*?)" (?P<status>\d{3}) (?P<bytes>\S+) "(?P<referrer>.*?)" "(?P<user_agent>.*?)"'
)


def get_log_files():
    search_path = os.path.join(LOG_DIR, LOG_FILENAME_PATTERN)
    files = glob.glob(search_path)
    if not files:
        print(f"Error: No log files found matching {search_path}")
        sys.exit(1)
    return sorted(files)


def parse_request_string(req_str):
    """
    Safely breaks down 'GET /path?q=1 HTTP/1.1'
    Returns: method, path, params
    """
    parts = req_str.split()

    method = "UNKNOWN"
    full_path = "-"

    if len(parts) > 0:
        method = parts[0]
    if len(parts) > 1:
        full_path = parts[1]

    if "?" in full_path:
        path, params = full_path.split("?", 1)
    else:
        path, params = full_path, "-"

    return method, path, params


def process_line_summary(
    line, ip_stats, global_counters, filter_code=None, debug=False
):
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            return

    match = LOG_PATTERN.search(line)
    if match:
        status = match.group("status")

        # Filter by Code if requested
        if filter_code and status != filter_code:
            return

        ip = match.group("ip")
        request_str = match.group("request_str")
        method, _, _ = parse_request_string(request_str)

        date_str = match.group("date").split(" ")[0]
        try:
            timestamp = datetime.strptime(date_str, "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            if debug:
                print(f"DEBUG: Failed Date Parse -> {date_str}")
            return

        stats = ip_stats[ip]
        stats["count"] += 1
        stats["methods"][method] += 1
        global_counters["total"] += 1

        if stats["first"] is None or timestamp < stats["first"]:
            stats["first"] = timestamp
        if stats["last"] is None or timestamp > stats["last"]:
            stats["last"] = timestamp
    else:
        if debug and line.strip():
            print(f"DEBUG: Failed Regex -> {line.strip()[:50]}...")


def process_line_query(line, target_ip, found_requests, filter_code=None, debug=False):
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            return

    # Optimization: if IP filter is on, quick check before regex
    if target_ip and target_ip not in line:
        return

    match = LOG_PATTERN.search(line)
    if match:
        # Strict IP check
        if target_ip and match.group("ip") != target_ip:
            return

        status = match.group("status")

        # Filter by Code
        if filter_code and status != filter_code:
            return

        date_str = match.group("date")
        try:
            timestamp = datetime.strptime(date_str.split(" ")[0], "%d/%b/%Y:%H:%M:%S")
        except ValueError:
            return

        request_str = match.group("request_str")
        method, path, params = parse_request_string(request_str)

        found_requests.append(
            {
                "ip": match.group(
                    "ip"
                ),  # Needed if no target_ip is provided (global search mode)
                "date": date_str,
                "timestamp": timestamp,
                "method": method,
                "path": path,
                "params": params,
                "status": status,
                "ua": match.group("user_agent"),
            }
        )
    else:
        if debug and line.strip():
            print(f"DEBUG: Failed Regex -> {line.strip()[:50]}...")


def parse_logs(target_ip=None, filter_code=None, debug=False):
    log_files = get_log_files()

    ip_stats = defaultdict(
        lambda: {"count": 0, "first": None, "last": None, "methods": defaultdict(int)}
    )
    global_counters = {"total": 0}
    found_requests = []

    print(f"Scanning {len(log_files)} log files...")
    if filter_code:
        print(f"Filtering for Status Code: {filter_code}")

    for file_path in log_files:
        try:
            if file_path.endswith(".gz"):
                f = gzip.open(file_path, "rb")
            else:
                f = open(file_path, "r", errors="ignore")

            with f:
                for line in f:
                    # If looking for specific details (IP OR Code search where we want list of requests)
                    if target_ip:
                        process_line_query(
                            line, target_ip, found_requests, filter_code, debug
                        )
                    else:
                        # Summary mode
                        process_line_summary(
                            line, ip_stats, global_counters, filter_code, debug
                        )
        except Exception as e:
            print(f"  [!] Error reading {file_path}: {e}")

    return ip_stats, global_counters, found_requests


def print_summary(ip_stats, total_reqs):
    if total_reqs == 0:
        print("No requests found matching criteria.")
        return

    sorted_ips = sorted(ip_stats.items(), key=lambda x: x[1]["count"], reverse=True)

    fmt = "{:<18} | {:<8} | {:<30} | {:<16} | {:<16}"
    print("-" * 100)
    print(fmt.format("IP ADDRESS", "COUNT", "METHODS", "FIRST SEEN", "LAST SEEN"))
    print("-" * 100)

    for ip, data in sorted_ips:
        first_str = data["first"].strftime("%Y-%m-%d %H:%M") if data["first"] else "-"
        last_str = data["last"].strftime("%Y-%m-%d %H:%M") if data["last"] else "-"

        meth_list = sorted(data["methods"].items(), key=lambda x: x[1], reverse=True)
        methods_str = ", ".join([f"{m}({c})" for m, c in meth_list])
        if len(methods_str) > 29:
            methods_str = methods_str[:26] + "..."

        print(fmt.format(ip, data["count"], methods_str, first_str, last_str))
    print("-" * 100)
    print(f"TOTAL REQUESTS FOUND: {total_reqs}")


def print_detailed_list(requests, title_prefix="Activity Report"):
    if not requests:
        print(f"No records found.")
        return

    requests.sort(key=lambda x: x["timestamp"])

    print(f"\n{title_prefix}")
    print(f"Total Requests found: {len(requests)}")

    for i, req in enumerate(requests):
        print("=" * 80)
        print(f"Request #{i+1} at {req['date']}")
        print(f"  IP:         {req['ip']}")
        print(f"  Method:     {req['method']} (Status: {req['status']})")
        print(f"  Path:       {req['path']}")

        if req["params"] != "-":
            print(f"  Params:     {req['params']}")

        print(f"  User Agent: {req['ua']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nginx Log Analyzer")
    parser.add_argument("--ip", help="Filter by specific IP address", default=None)
    parser.add_argument(
        "--code", help="Filter by HTTP Status Code (e.g. 200, 404, 500)", default=None
    )
    parser.add_argument(
        "--debug", help="Print lines that fail regex parsing", action="store_true"
    )
    args = parser.parse_args()

    ip_stats, global_counters, found_requests = parse_logs(
        target_ip=args.ip, filter_code=args.code, debug=args.debug
    )

    # Logic for display:
    # 1. If IP is provided, show Detailed List
    if args.ip:
        print_detailed_list(found_requests, title_prefix=f"Report for IP: {args.ip}")
    # 2. If NO IP but just SUMMARY
    else:
        print_summary(ip_stats, global_counters["total"])